# 抽取关键帧
# 给定 GIF，返回 K 张关键帧的 PIL images

import io, os
import json
import numpy as np
import random
from functools import partial
from PIL import Image
from PIL import Image, ImageSequence
import imageio
import shutil
import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2
import operator
from sklearn.cluster import KMeans



def read_gifs(func):
    def wrapper(gif_path, k=4):
        img = Image.open(gif_path)
        frames = [i.copy() for i in ImageSequence.Iterator(img)]

        lens = len(frames)
        if lens == k: return frames, [i for i in range(k)]
        if lens == 1: return frames * k, [0 for i in range(k)]
        while lens < k:
            frames += frames[::-1][1:]
            lens = len(frames)

        res = func(frames, k=4)
        return res
    return wrapper

# 等间隔抽帧
@read_gifs
def extract_by_equal_space(frames, k=4):
    """
    frames: Pil 读取的 GIF
    k: 关键帧张数
    return 不同关键帧的 PIL images
    """
    # 抽帧
    lens = len(frames)
    idxs = np.linspace(0,lens-1,num=k,endpoint=True,retstep=False, dtype=None)
    idxs = list(map(round, idxs))
    return [frames[idx] for idx in idxs]

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
    # Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    # Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
    # Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
    # Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
    # It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out

# 基于特征聚类的抽帧方法
@read_gifs
def extract_by_feature_cluster(frames, k=4):
    """
    frames: Pil 读取的 GIF
    k: 关键帧张数
    return 不同关键帧的 PIL images
    """

    # 特征抽取
    frames_cv2 = [cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) for img in frames]
    features = []
    for frame in frames_cv2:
        frame = transform(frame)
        # print(frames.shape)
        frame = frame.reshape(-1, 3, 448, 448)

        img = frame.to(device)
        # We only extract features, so we don't need gradient
        with torch.no_grad():
            # Extract the feature from the image
            feature = new_model(img)
        # Convert to NumPy Array, Reshape it, and save it to features variable
        features.append(feature.cpu().detach().numpy().reshape(-1))
                            
    features = np.array(features) 

    # 聚类
    cluster_model = KMeans(n_clusters=k, random_state=42)
    cluster_model.fit(features)
    labels = cluster_model.labels_

    # 抽帧
    items = {i:0 for i in range(k)}
    idxs = []
    for idx, label in enumerate(labels):
        if label in items:
            items.pop(label)
            idxs.append(idx)
        if len(items.keys()) == 0:
            break

    return [frames[idx] for idx in idxs], idxs

def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    """
    print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
 
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x

# 基于帧间差法的抽帧方法
@read_gifs
def extract_by_interframe_difference(frames, k=4):
    """
    frames: Pil 读取的 GIF
    k: 关键帧张数
    return 不同关键帧的 PIL images
    """
    frames_cv2 = [cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) for img in frames]


    curr_frame = None
    prev_frame = None 
    frame_diffs = []
    frames_tmp = []
    i = 0 
    for frame in frames_cv2:
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            #logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames_tmp.append(frame)
        prev_frame = curr_frame
        i = i + 1

    # compute keyframe
    keyframe_id_set = set()
    # sort the list in descending order
    frames_tmp.sort(key=operator.attrgetter("diff"), reverse=True)
    for keyframe in frames_tmp[:k]:
        keyframe_id_set.add(keyframe.id) 
    
    print(sorted(list(keyframe_id_set)))
    return [frames[idx] for idx in sorted(list(keyframe_id_set))]



if __name__=='__main__':
    test_cases = [
        '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/i2vgen-xl-diffusers/test_dir/test_case_gif/8f624919fe514232a5abb4a45de3f10a.gif',
        '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/i2vgen-xl-diffusers/test_dir/test_case_gif/4e542f11aa3f5bf4510d96e934ba98e8.gif'
    ]
    method = 'extract_by_equal_space'

    # prepare
    if method == 'extract_by_feature_cluster':
        model = models.vgg16(pretrained=True)
        new_model = FeatureExtractor(model)
        # Change the device to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        new_model = new_model.to(device)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(512),
            transforms.Resize(448),
            transforms.ToTensor()                              
            ])


    for test_case in test_cases:
        frames = eval(method)(test_case)
        print(frames)
        _, filename = os.path.split(test_case)
        frames[0].save(f'key_{method}_{filename}', save_all=True, append_images=frames[1:], disposal=2)
        print(f'Keyframes save to {filename} .')

        shutil.copy(
            test_case,
            f'src_{filename}'
        )
        print(f'Srcframes save to {filename} .')