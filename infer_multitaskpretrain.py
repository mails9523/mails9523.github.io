import torch
# from diffusers import I2VGenXLPipeline
from vldm.pipeline_multitaskpretrain import DFGNPipeline
from diffusers.utils import load_image, export_to_gif
import ipdb
import shutil
import os, json, random
# from vldm.unet_multitaskpretrain import I2VGenXLUNet
from diffusers.loaders import LoraLoaderMixin
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.attention import Attention, FeedForward
from lmdb_dataset_v2 import DatasetForFrameInterpolation 
import numpy as np
from PIL import Image, ImageSequence

from vldm.DFGN import DFGN
from train_multitaskpretrain import TaskRandomChosen
from lmdb_dataset_v2 import DatasetForMultitaskPretrain as Dataset

# unet 权重路径
unet_ckpt_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zhangting/code/i2v_0606/i2vgen-dev/sd-model-finetuned/logs_multitaskpretrain'
unet_ckpt = 'checkpoint-60000'
# 其余模型组件加载路径
base_model_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/data/torch_cache/ali-vilab-i2vgen-xl/models--ali-vilab--i2vgen-xl/snapshots/39e1979ea27be737b0278c06755e321f2b4360d5'

paths = {
    'unet': os.path.join(unet_ckpt_path, unet_ckpt),
    'vae': base_model_path,
    'T5_tokenizer': base_model_path,
    'clip_tokenizer': base_model_path,
    'T5_text_encoder': base_model_path,
    'clip_text_encoder': base_model_path,
    'scheduler': base_model_path
}

pipeline = DFGNPipeline.from_pretrained(paths).to("cuda")


import datetime
def load_from_json(json_path):
    data = []
    with open(json_path) as fin:
        for l in fin:
            data.append(json.loads(l))
    return data

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(6666)

# /usr/local/envs/diffusers/bin/python

def gene_time_str():
    time1 = datetime.datetime.now()
    time1 = time1.strftime('%Y_%m_%d_%H_%M_%S')
    return time1

# 指定任务/目录
task_name = 'interpolation' # {'i2v', 't2v', 'it2v', 'interpolation', 'pre_frame_prediction', 'post_frame_prediction'}
infer_or_eval = 'eval' # {'train':json文件中读取测试数据, 'eval':本地读取测试数据}
exp_name = 'multitask'
infer_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zhangting/code/i2v_0606/i2vgen-dev/test_dir'
infer_dir = 'k8_pretrain_all_0621'

size = 256
rank = 32
num_frames= 8

exp_name = '{}_{}_{}_{}_{}_rank_{}_{}'.format(exp_name, unet_ckpt, task_name, gene_time_str(), size, rank, infer_or_eval)
os.makedirs(os.path.join(infer_path, "{}_{}".format(exp_name, size)), exist_ok=True)

pipeline.unet.eval()
pipeline.enable_model_cpu_offload()

dataset = Dataset()
trc = TaskRandomChosen()

if infer_or_eval == 'train':
    infer_json_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/dataset_cartoon_v2/new_data/k8_pretrain_all.json'
    infer_json = load_from_json(infer_json_path)
    lens = len(infer_json)
    infer_json = [infer_json[random.randint(0, lens)] for i in range(50)]

    for idx, item in enumerate(infer_json):
        input_text = item['llava_wx_chn_cap'] if 'llava_wx_chn_cap' in item else item['eng_cap'][:120]
        name = item['md5']
        keyframe_indexs = item['keyframe_idx']

        # print(item['visual_path'])

        imgs = Image.open(item['visual_path'])

        frames = []
        for sketch in ImageSequence.Iterator(imgs):
            frames.append(sketch.copy().resize((size, size)))
        try:
            frames = [frames[idx] for idx in keyframe_indexs]
        except:
            continue
        if len(frames) > num_frames:
            frames = frames[:num_frames]
        
        # 创建任务名称及对应mask的字典
        min_num_frames = len(frames)
        task_mask_dict = {
            'i2v': trc.make_condition('i2v', min_num_frames),
            't2v': trc.make_condition('t2v', min_num_frames),
            'it2v': trc.make_condition('it2v', min_num_frames),
            'interpolation': trc.make_condition('interpolation', min_num_frames),
            'pre_frame_prediction': trc.make_condition('pre_frame_prediction', min_num_frames),
            'post_frame_prediction': trc.make_condition('post_frame_prediction', min_num_frames)
        }

        # 获取当前任务的文本信息(bool)和结构控制条件(list)
        cur_task_condition = task_mask_dict.get(task_name)
        wo_text = cur_task_condition['text_drop']
        info_mask = cur_task_condition['info_mask']
        
        # input_tensor: 输入的tensor(目前和训练阶段对齐，都补齐到8帧);
        # refer_images: mask后的参考图/视频;
        # all_images: 真值; frame_mask: 帧mask
        input_tensor, refer_images, all_images, frame_mask = dataset.embeddings_testgif(frames, info_mask)

        negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        generator = torch.manual_seed(6666)

        frames = pipeline(
            prompt=input_text,
            image=input_tensor,
            info_mask = info_mask,
            frame_mask = frame_mask,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
            height = size,
            width = size,
            num_frames = num_frames,
            wo_text = wo_text,
        ).frames[0]

        video_path = export_to_gif(frames, os.path.join(infer_path,  "{}_{}".format(exp_name, size), name[:-4]+'_pred.gif'))
        src_path = export_to_gif(refer_images, os.path.join(infer_path,  "{}_{}".format(exp_name, size), name[:-4]+'_ref.gif'))
        gt_path = export_to_gif(all_images, os.path.join(infer_path,  "{}_{}".format(exp_name, size), name[:-4]+'_all.gif'))
        
        print("{}/{} done".format(idx, len(os.listdir(os.path.join(infer_path, infer_dir)))))
        print(f'save at {exp_name}_{size}')

elif infer_or_eval == 'eval':

    # for idx, name in enumerate(os.listdir(os.path.join(infer_path, infer_dir))):
    #     input_text = name[:-4]
    #     file_path = os.path.join(infer_path, infer_dir, name)

    files = os.listdir(os.path.join(infer_path, infer_dir))
    file_pairs = {}

    # 遍历文件，找到成对gif和txt文件
    for file in files:
        if file.endswith('.gif') or file.endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            if base_name not in file_pairs:
                file_pairs[base_name] = {}
            if file.endswith('.gif'):
                file_pairs[base_name]['gif'] = file
            elif file.endswith('.txt'):
                file_pairs[base_name]['txt'] = file

    for base_name, pair in file_pairs.items():
        if 'gif' in pair and 'txt' in pair:
            gif_path = os.path.join(os.path.join(infer_path, infer_dir), pair['gif'])
            txt_path = os.path.join(os.path.join(infer_path, infer_dir), pair['txt'])
            
            # 读取当前输入（图像/视频）
            img = Image.open(gif_path)
            frames = []
            for sketch in ImageSequence.Iterator(img):
                frames.append(sketch.copy().resize((size, size)))
            if len(frames) > num_frames:
                frames = frames[:num_frames]
            
            # 创建任务名称及对应mask的字典
            min_num_frames = len(frames)
            task_mask_dict = {
                'i2v': trc.make_condition('i2v', min_num_frames),
                't2v': trc.make_condition('t2v', min_num_frames),
                'it2v': trc.make_condition('it2v', min_num_frames),
                'interpolation': trc.make_condition('interpolation', min_num_frames),
                'pre_frame_prediction': trc.make_condition('pre_frame_prediction', min_num_frames),
                'post_frame_prediction': trc.make_condition('post_frame_prediction', min_num_frames)
            }

            # 获取当前任务的文本信息(bool)和结构控制条件(list)
            cur_task_condition = task_mask_dict.get(task_name)
            wo_text = cur_task_condition['text_drop']
            info_mask = cur_task_condition['info_mask']

            if wo_text:
                input_text = ''
            else:
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    input_text = txt_file.read()
                    print(f"Content of {base_name}:\n{input_text}\n")
                save_text_file_path = os.path.join(infer_path,  "{}_{}".format(exp_name, size), base_name+'.txt')
                with open(save_text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(input_text)
            
            # input_tensor: 输入的tensor(目前和训练阶段对齐，都补齐到8帧);
            # refer_images: mask后的参考图/视频;
            # all_images: 真值; frame_mask: 帧mask
            input_tensor, refer_images, all_images, frame_mask = dataset.embeddings_testgif(frames, info_mask)

            negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
            generator = torch.manual_seed(6666)

            frames = pipeline(
                prompt=input_text,
                image=input_tensor,
                info_mask = info_mask,
                frame_mask = frame_mask,
                num_inference_steps=50,
                negative_prompt=negative_prompt,
                guidance_scale=9.0,
                generator=generator,
                height = size,
                width = size,
                num_frames = num_frames,
                wo_text = wo_text,
            ).frames[0]

            video_path = export_to_gif(frames, os.path.join(infer_path,  "{}_{}".format(exp_name, size), base_name+'_pred.gif'))
            src_path = export_to_gif(refer_images, os.path.join(infer_path,  "{}_{}".format(exp_name, size), base_name+'_ref.gif'))
            gt_path = export_to_gif(all_images, os.path.join(infer_path,  "{}_{}".format(exp_name, size), base_name+'_all.gif'))
            
            # print("{}/{} done".format(idx, len(os.listdir(os.path.join(infer_path, infer_dir)))))
            print(f'save at {exp_name}_{size}')



# if __name__=="__main__":

#     def load_from_json(json_path):
#         data = []
#         with open(json_path) as fin:
#             for l in fin:
#                 data.append(json.loads(l))
#         return data

#     # 储存train json中的测试图像 用于 infer_or_eval == 'eval'
#     save_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zhangting/code/i2v_0606/i2vgen-dev/test_dir/k8_pretrain_all_0621'
#     size = 256
#     infer_json_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/dataset_cartoon_v2/new_data/k8_pretrain_all.json'
#     infer_json = load_from_json(infer_json_path)
#     lens = len(infer_json)
#     infer_json = [infer_json[random.randint(0, lens)] for i in range(100)]
#     # i = 0
#     for idx, item in enumerate(infer_json):
#         input_text = item['llava_wx_chn_cap'] if 'llava_wx_chn_cap' in item else item['eng_cap'][:120]
#         name = item['md5']
#         keyframe_indexs = item['keyframe_idx']
        
#         imgs = Image.open(item['visual_path'])
#         frames = []
#         for sketch in ImageSequence.Iterator(imgs):
#             frames.append(sketch.copy().resize((size, size)))
        
#         all_images = []
#         for f in frames:
#             tmp = f.copy()
#             if tmp.mode != 'RGBA':
#                 tmp = tmp.convert('RGBA')
#             image = Image.new('RGB', size=(size, size), color=(255, 255, 255))
#             image.paste(tmp, (0, 0), mask=tmp)
#             all_images.append(image)

#         gt_path = export_to_gif(all_images, os.path.join(save_path, name+'.gif'))
#         text_file_path = os.path.join(save_path, name + '.txt')
#         with open(text_file_path, 'w', encoding='utf-8') as text_file:
#             text_file.write(input_text)


#         # print(item['visual_path'])
