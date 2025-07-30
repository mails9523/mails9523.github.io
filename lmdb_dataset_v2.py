import io, os
import json
import numpy as np
import random
import lmdb
import torch
from functools import partial
from PIL import Image
from PIL import Image, ImageSequence
import imageio
import cv2

from torch.utils import data
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms as T

# from get_condition import get_binary_img, get_contour, get_sketch_by_cv2
import ipdb
import jieba
import numpy as np
import random
seed = 6666
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def preprocess_img(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    img = img.copy().convert('RGBA')
    img = img.resize((256, 256))
    bg = Image.new('RGBA', (256, 256), (255,255,255,255))
    bg.paste(img, (0, 0), mask=img)
    bg = bg.convert(mode)
    return bg

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    for frame in ImageSequence.Iterator(img):
        frame = frame.copy().convert('RGBA')
        bg = Image.new('RGBA', (256, 256), (255,255,255,255))
        bg.paste(frame, (8, 8), mask=frame)
        yield bg.convert(mode)

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def frames_to_tensor(frames, channels = 3, transform = T.ToTensor()):
    tensors = [transform(preprocess_img(f, channels = channels)) for f in frames]
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

# def cast_num_frames(t, *, frames):
#     f = t.shape[1]
#     if f == frames:
#         return t
#     if f > frames:
#         return t[:, :frames]
#     return torch.cat((t, t[:, :frames-f]), dim=1)
#     # return F.pad(t, (0, 0, 0, 0, 0, frames - f))

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f == frames:
        return t
    # if f > frames:
    #     idxs = np.linspace(0,f-1,num=frames,endpoint=True,retstep=False, dtype=None)
    #     idxs = list(map(round, idxs))
    #     res = torch.cat([t[:, idx].unsqueeze(dim=1) for idx in idxs], dim=1)
    #     return res

        # return t[:, :frames]

    while f < frames:
        t = torch.cat((t, t[:, :frames-f]), dim=1)
        f = t.shape[1]
    return t[:, :frames]
    # return F.pad(t, (0, 0, 0, 0, 0, frames - f))



class Dataset(data.Dataset):
    def __init__(
        self,
        json_file,
        image_size,
        channels = 3,
        num_frames = 8,
        # horizontal_flip = False,
        force_num_frames = True,
        need_sketch = True,
        llava_cap = False,
        mean_std = [[0.894, 0.864, 0.839], [0.254, 0.279, 0.310]]
    ):
        super().__init__()
        self.data = [i for i in self.load_data(json_file) if i['fmt']== 'gif']
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5]),
            T.Normalize(mean_std[0], mean_std[1]),
    ])
        self.transform_sketch = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor()
    ])
        self.need_sketch = need_sketch
        self.llava_cap = llava_cap
        
        # if self.llava_cap:
        #     self.llava_map = {}
        #     with open('/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/video_diffusion_yzq/LLaVA/all_translate/llava_cn_to_jp.txt') as fin:
        #         for l in fin:
        #             md5_, cap_ = l.replace('\n', '').split('\t')
        #             self.llava_map[md5_] = cap_
            

    def load_data(self, file):
        data = []
        with open(file) as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # if self.llava_cap and item['md5'] in self.llava_map:
        #     caption = item['meanings'] + ', ' + self.llava_map[item['md5']]
        # else:
        #     caption = item['meanings'] + ', ' + item['caption']
        caption = item['eng_cap']
        
        frames = self.get_frames(item['md5'], item['nframes'])
        #  tensor = gif_to_tensor(item['img_path'], self.channels, transform = self.transform)
        tensor = frames_to_tensor(frames, transform = self.transform)
        tensor = self.cast_num_frames_fn(tensor)
        # get sketch
        if self.need_sketch:
            sketchs = self.get_sketch(frames, self.transform_sketch, self.num_frames)
            return {'caption': caption, 'tensor': tensor, 'sketch': sketchs}
        else:
            return {'caption': caption, 'tensor': tensor}

    def get_frames(self, md5, nframes):
        frames = []
        for idx in range(min(nframes, self.num_frames)):
            key = f'{md5}_{idx}'
            frame = frames_txn.get(key.encode())
            frame = Image.open(io.BytesIO(frame))
            frames.append(frame)
        return frames

    def get_sketch(self, nframes, transform, len_frame, cv2_extract=True):

        while len(nframes) <= len_frame:
            nframes += nframes

        sketchs = []
        for frame in nframes[:len_frame]:
            gray_img = np.array(frame.convert('L'))
            if cv2_extract:
                contour_img = get_sketch_by_cv2(gray_img)
            else:
                bin_img = get_binary_img(gray_img)
                contour_img = get_contour(bin_img)
            contour_img = Image.fromarray(contour_img.astype('uint8'))
            sketchs.append(contour_img)
        # to tensor
        tensors = tuple(map(transform, sketchs))
        return torch.stack(tensors, dim = 1)


def infer_input_sketch_data_generation(condition_sketch_path, image_size=256, num_frames=8, cv2_extract=True):
    transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5]),
    ])

    all_tensors = []
    for p in condition_sketch_path:
        frames=imageio.mimread(p)
        while len(frames) < num_frames:
            frames += frames
        sketchs = []
        for frame in frames[:num_frames]:
            frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            gray_img = np.array(frame.convert('L'))
            if cv2_extract:
                contour_img = get_sketch_by_cv2(gray_img)
            else:
                bin_img = get_binary_img(gray_img)
                contour_img = get_contour(bin_img)
            contour_img = Image.fromarray(contour_img.astype('uint8'))
            sketchs.append(contour_img)
        tensors = tuple(map(transform, sketchs))
        tmp_tensor = torch.stack(tensors, dim = 1)
        all_tensors.append(tmp_tensor)
    all_tensors = torch.stack(all_tensors)
    all_tensors = all_tensors.to(memory_format=torch.contiguous_format).float()
    return all_tensors

def save_sketch_tensor_to_gif(condition_sketch, save_path):
    for idx in range(len(condition_sketch)):
        tmp = condition_sketch[idx][0].numpy()* 255
        out_gif = []
        for frame_idx in range(len(tmp)):
            tmp_1 = tmp[frame_idx]
            tmp_1[random.randint(0,len(tmp)-1)][random.randint(0,len(tmp[0])-1)] = random.randint(0,254)
            frame = Image.fromarray(tmp_1.astype('uint8')).convert('RGB')
            out_gif.append(frame)
                    # Image.save(frame, os.path.join(save_path, '{}_{}.'))

        out_gif[0].save(f'{save_path}/{idx}.gif', save_all=True, append_images=out_gif[1:], disposal=2)



class DatasetForMultitaskPretrain(data.Dataset):
    def __init__(
        self,
        json_file='/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/dataset_cartoon_v2/new_data/k8_pretrain_all.json',
        image_size=256,
        channels = 3,
        force_num_frames = True,
        text_drop_prob = 0.2,
        num_frames = 8,
        image_dir = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/i2vgen-xl-diffusers/tools/cache'
    ):

        super().__init__()
        self.data = self.load_data(json_file)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.text_drop_prob = text_drop_prob

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

        self.image_dir = image_dir

    def load_data(self, file):
        data = []
        with open(file) as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    def __len__(self):
        return len(self.data)

    import random
    def __getitem__(self, index):
        while True:
            item = self.data[index]

            keyframe_indexs = item['keyframe_idx']

            # get frames
            img = Image.open(os.path.join(self.image_dir, item['visual_path'].split('/')[-1]))

            src_frames = []
            
            try:
                for sketch in ImageSequence.Iterator(img):
                        src_frames.append(sketch.copy())
            except:
                index = random.randint(0, self.__len__()-1)
                continue    

            frames = []
            for idx in keyframe_indexs:
                try:
                    frames.append(src_frames[idx])
                except:
                    pass

            # frames = [frames[idx] ]

            tensor = frames_to_tensor(frames, transform = self.transform)
            tensor = 2.0 * tensor - 1.0

            # support more frame nums
            if len(frames) != self.num_frames:
                c, f, h, w = tensor.shape
                tensor = torch.cat([tensor, torch.zeros(c, self.num_frames - f, h, w)], dim=1)

            # get loss mask
            frame_mask = [1] * len(frames) + [0] * (self.num_frames - len(frames))

            # get caption
            caption = item['llava_wx_chn_cap'] if 'llava_wx_chn_cap' in item else item['eng_cap']
            caption = self.text_argu(caption, drop_prob=self.text_drop_prob)

            return {'caption': caption, 'tensor': tensor, 'frame_mask': frame_mask}

    def text_argu(self, caption, drop_prob = 0.2):
        iteration = jieba.cut(caption)
        iteration = map(lambda x: '#' if random.randint(0,100) <= int(100 * drop_prob) and x != ' ' else x, iteration)
        results = ''.join(iteration)
        return results
    
    def embeddings_testgif(self, frames, info_mask, only_first=False):
        # frames:  gif图像列表
        # info_mask: 不同任务的掩码信息 eg.[1,0,0,0,0,0,0,0]

        for f in frames:
            tmp = f.copy()
            if tmp.mode != 'RGBA':
                tmp = tmp.convert('RGBA')
            image = Image.new('RGB', size=(self.image_size, self.image_size), color=(255, 255, 255))
            image.paste(tmp, (0, 0), mask=tmp)

        tensor = frames_to_tensor(frames)
        tensor = 2.0 * tensor - 1.0
        # print(len(frames))

        if len(frames) != self.num_frames:
            c, f, h, w = tensor.shape
            tensor = torch.cat([tensor, torch.zeros(c, self.num_frames - f, h, w)], dim=1)

        frame_mask = [1] * len(frames) + [0] * (self.num_frames - len(frames))
        
        # all_images: gif的真值
        all_images = []
        for f in frames:
            tmp = f.copy()
            if tmp.mode != 'RGBA':
                tmp = tmp.convert('RGBA')
            image = Image.new('RGB', size=(self.image_size, self.image_size), color=(255, 255, 255))
            image.paste(tmp, (0, 0), mask=tmp)
            all_images.append(image)

        # refer_images: 输入unet的参考图像
        refer_images = []
        for i in range(len(all_images)):
            if info_mask[i] == 1:
                refer_images.append(all_images[i].copy())
            else:
                refer_images.append(Image.new('RGB', all_images[i].size, (0, 0, 0)))

        return tensor, refer_images, all_images, frame_mask


class DatasetForFirstFrameGuide(data.Dataset):
    # TODO:
    # 1. 过滤少帧
    # 2. 记录及删除所有open出错的md5
    # 3. 文本处理这块
    # 4. 和 zt 统一图像处理的 tensor  
    # out_fmt = {
    #   "caption": 'text',
    #   "tensor": torch.tensor, 
    #   "sketch": PIL Image
    # }
    def __init__(
        self,
        json_file,
        image_size=256,
        channels = 3,
        num_frames = 8,
        force_num_frames = True,
        llava_cap = False,
        mean_std = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]], 
        image_path = '',
        sketch = True
    ):

        super().__init__()
        self.data = self.load_data(json_file)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5]),
            # T.Normalize(mean_std[0], mean_std[1]),
    ])


        self.llava_cap = llava_cap
        if self.llava_cap:
            self.llava_map = {}
            with open('/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/video_diffusion_yzq/LLaVA/all_translate/llava_cn_to_jp.txt') as fin:
                for l in fin:
                    md5_, cap_ = l.replace('\n', '').split('\t')
                    self.llava_map[md5_] = cap_
            
        self.caption = None
        self.tensor = None
        self.sketch = None
        self.image_path = image_path
        self.sketch = sketch


    def load_data(self, file):
        data = []
        with open(file) as fin:
            for l in fin:
                tmp = json.loads(l)
                if len(set(tmp['keyframe_idx'])) < 4:
                    continue
                data.append(tmp)
        data = [i for i in data if i['fmt']== 'gif']
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            item = self.data[index]

            keyframe_indexs = item['keyframe_idx']
            # 去除帧数少的干扰
            # if len(set(keyframe_indexs)) < 4 and self.tensor != None:
                # return {'caption': self.caption, 'tensor': self.tensor, 'sketch': self.sketch}

            # get sketch
            try:
                img = Image.open(item['visual_path'])
                frames = []
                for sketch in ImageSequence.Iterator(img):
                    frames.append(sketch.copy())

                frames = [frames[idx] for idx in keyframe_indexs]
                # frames = list(map(Lambda x:frames[x], keyframe_indexs))
            except:
                index = random.randint(0, self.__len__())
                continue

            caption = item['llava_wx_chn_cap'] if 'llava_wx_chn_cap' in item else item['eng_cap']

            tensor = frames_to_tensor(frames, transform = self.transform)
            # tensor = self.cast_num_frames_fn(tensor)
            tensor = 2.0 * tensor - 1.0

            sketch = frames[0].copy().convert('RGB')

            self.caption = caption
            self.tensor = tensor
            sketch = sketch if not self.sketch else self.get_sketch_by_cv2(sketch)
            
            # print(tensor.shape)

            return {'caption': caption, 'tensor': tensor, 'sketch': sketch}
            # except:
            #     print("Error image: {}".format(item['visual_path']))
            #     return {'caption': self.caption, 'tensor': self.tensor, 'sketch': self.sketch}

    def get_frames(self, md5, nframes, gif_path):
        frames = []
        for idx in range(min(nframes, self.num_frames)):
            # key = f'{md5}_{idx}'
            # frame = frames_txn.get(key.encode())
            frame = Image.open(gif_path)
            frames.append(frame)
        return frames
    
    
    def get_sketch_by_cv2(self, img):
        gray_img = np.array(img.convert('L'))
        edge = cv2.Canny(gray_img, 20, 100)
        # edge = np.array(edge * (edge>125), dtype=np.uint8)
        edge = np.array(edge, dtype=np.uint8)
        contour_img = Image.fromarray(edge.astype('uint8')).convert('RGB')
        return contour_img



def infer_input_random_frame_generation(condition_sketch_path):
    all_images = []
    for p in condition_sketch_path:
        frames=imageio.mimread(p)
        frame = frames[random.randint(0, len(frames)-1)]
        frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
        all_images.append(frame)
    return all_images

def save_random_frame_to_jpg(condition_sketch, save_path):
    for idx, frame in enumerate(condition_sketch):
        Image.Image.save(frame, os.path.join(save_path, '{}.jpg'.format(idx)))




class DatasetForFrameInterpolation(data.Dataset):
    def __init__(
        self,
        json_file='/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/dataset_cartoon_v2/new_data/data_v4_fine_append_ysh.json',
        image_size=256,
        channels = 3,
        num_frames = 4,
        force_num_frames = True,
        llava_cap = False,
        mean_std = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]], 
        image_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/jiapeizhang/from_apdcephfs/emoticon_inpainting.v2/images',
        train_mask = True
    ):

        super().__init__()

        self.data = self.load_data(json_file)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.train_mask = train_mask

        self.cast_num_frames_fn = partial(cast_num_frames, frames = 2*num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize([0.5], [0.5]),
            # T.Normalize(mean_std[0], mean_std[1]),
        ])
        self.transform_test = T.Compose([
        T.Resize(image_size),
        # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
        #  T.CenterCrop(image_size),
        T.ToTensor(),
        # T.Normalize([0.5], [0.5]),
])
        self.caption_cache = None
        self.inputs_cache = None
        self.pil_images_cache = None
        self.outputs_cache = None
        self.image_path = image_path

        self.input_idx = list(range(2*num_frames))[0:2*num_frames:2]
        self.output_idx = list(range(2*num_frames+1))[1:2*num_frames:2]

    
    def load_data(self, file):
        data = []
        with open(file) as fin:
            for idx, l in enumerate(fin):
                # if idx>=30000: break
                data.append(json.loads(l))
        data = [i for i in data if i['fmt']== 'gif']
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:

            item = self.data[index]

            caption = item['llava_wx_chn_cap'] if 'llava_wx_chn_cap' in item else item['eng_cap']

            # '{}/{}.{}'.format(self.image_path, item['md5'], item['fmt'])

            # get images
            try:
                img = Image.open(item['visual_path'])
                frames = []
                for sketch in ImageSequence.Iterator(img):
                    frames.append(sketch.copy())

                keyframe_indexs = item['keyframe_idx']
                frames = [frames[idx] for idx in keyframe_indexs]
            except:
                index = random.randint(0, self.__len__())
                continue

            # 去除帧数少的干扰
            if len(set(keyframe_indexs)) < 4 and self.inputs_cache != None:
                return {'inputs': self.inputs_cache, 'outputs': self.outputs_cache, 'caption': self.caption_cache, 'pil_images':self.pil_images_cache}

            # while len(frames) < (2* self.num_frames + 1):
            #     if len(frames) == 1: frames = frames + frames
            #     frames = frames + frames[::-1][1:]
                # print(len(frames))
            frames = frames[:2* self.num_frames]

            tensor = frames_to_tensor(frames, transform = self.transform)
            tensor = 2.0 * tensor - 1.0
            # tensor = self.cast_num_frames_fn(tensor)
            
            # inputs & pil image & outputs
            inputs = torch.cat([tensor[:, idx].unsqueeze(dim=1) for idx in self.input_idx], dim=1)
            pil_images = [frames[idx].copy().convert('RGB') for idx in self.input_idx]
            outputs = torch.cat([tensor[:, idx].unsqueeze(dim=1) for idx in self.output_idx], dim=1)


            if self.train_mask:
                # random_input = [inputs[:, 0]]
                for idx in range(inputs.shape[1]-1):
                    cond = random.randint(0,1)
                    if cond:
                        # random_input.append(inputs[:, idx+1])
                        inputs[:, idx+1] = inputs[:, idx+1] * 0
                    # else:
                        # random_input.append(inputs[:, idx+1] * 0)
                # inputs = torch.cat(random_input, dim=1)

            self.caption_cache = caption
            self.inputs_cache = inputs
            self.outputs_cache = outputs
            self.pil_images_cache = pil_images

            # print({'inputs': inputs, 'outputs': outputs, 'caption': caption, 'pil_images': pil_images})

            return {'inputs': inputs, 'outputs': outputs, 'caption': caption, 'pil_images': pil_images}
            # except Exception as e:
            #     print(e)
            #     print("Error image: {}".format(item['visual_path']))
            #     return {'inputs': self.inputs_cache, 'outputs': self.outputs_cache, 'caption': self.caption_cache, 'pil_images':self.pil_images_cache}    

    def embeddings_testgif(self, gif_path, only_first=False, infer=False):
        img = Image.open(gif_path)
        frames = []
        for sketch in ImageSequence.Iterator(img):
            frames.append(sketch.copy().resize((self.image_size, self.image_size)))
        
        if not infer:
            while len(frames) < (2* self.num_frames + 1):
                if len(frames) == 1: frames = frames + frames
                frames = frames + frames[::-1][1:]
            frames = frames[:2* self.num_frames]
        
        # print('len(frames):',len(frames))
        tensor = frames_to_tensor(frames, transform = self.transform_test)

        inputs = 2.0 * tensor - 1.0

        if not infer:
            inputs = torch.cat([inputs[:, idx].unsqueeze(dim=1) for idx in self.input_idx], dim=1)
        

        pil_images = []
        for idx in range(self.num_frames):
            if not infer:
                tmp = frames[self.input_idx[idx]].copy()
            else:
                tmp = frames[idx].copy()

            if tmp.mode != 'RGBA':
                tmp = tmp.convert('RGBA')
            image = Image.new('RGB', size=(self.image_size, self.image_size), color=(255, 255, 255))
            image.paste(tmp, (0, 0), mask=tmp)
            pil_images.append(image)

        all_images = []
        for f in frames:
            tmp = f.copy()
            if tmp.mode != 'RGBA':
                tmp = tmp.convert('RGBA')
            image = Image.new('RGB', size=(self.image_size, self.image_size), color=(255, 255, 255))
            image.paste(tmp, (0, 0), mask=tmp)
            all_images.append(image)

        if only_first:
            for idx in range(inputs.shape[1]-1):
                inputs[:, idx+1] = inputs[:, idx+1] * 0

        return inputs, pil_images, tensor, all_images


class DataseTextToSketch(data.Dataset):
    def __init__(
        self,
        json_file,
        image_size,
        channels = 3,
        num_frames = 8,
        # horizontal_flip = False,
        force_num_frames = True,
    ):
        super().__init__()
        self.data = self.load_data(json_file)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity
        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            #  T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
    ])

    def load_data(self, file):
        data = []
        with open(file) as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        caption = item['meanings'] + ', ' + item['caption']
        frames = self.get_frames(item['md5'], item['nframes'])
        #  tensor = gif_to_tensor(item['img_path'], self.channels, transform = self.transform)
        tensor = frames_to_tensor(frames, transform = self.transform)
        tensor = self.cast_num_frames_fn(tensor)

        # generate 0
        zeros = torch.zeros(3, 1, 256, 256) - 1
        tensors = torch.cat([zeros, tensor], dim=1)

        return {'caption': caption, 'tensor': tensor, 'sketch': tensors[:, :-1, :, :]}

    def get_frames(self, md5, nframes):
        frames = []
        for idx in range(min(nframes, self.num_frames)):
            key = f'{md5}_{idx}'
            frame = frames_txn.get(key.encode())
            frame = Image.open(io.BytesIO(frame)).convert('L')
            frame = np.array(frame)
            frame = get_sketch_by_cv2(frame)
            frame = np.expand_dims(frame, axis=-1)
            frame = np.c_[frame, frame, frame]
            frame = Image.fromarray(frame.astype('uint8'))
            frames.append(frame)
        return frames


if __name__ == "__main__":
    datasets = DatasetForFrameInterpolation()
    ipdb.set_trace()
    a = datasets.__getitem__(0)


# for step, batch in enumerate(train_dataloader):
#     print(batch)
#     break