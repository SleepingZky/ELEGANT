#coding=utf-8
########################################
# 本文件是对于 ELEGANT 论文的复现
# 因为 ELEGANT 的实现看起来比较简单一些
# 所以就挑 ELEGANT 入手
########################################

import numpy as numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms utils
import os,time 
from PIL import Image


# 该文件主要是用来进行基础的配置
class Config:
    @property
    def data_dir(self):
        data_dir = './dataset/celebA'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join('train_log')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir
        
    @property
    def model_dir(self):
        return 

    @property
    def log_dir(self):
        return 

    @property
    def img_dir(self):
        return 

    # 定义了一些超参数
    nchw=[16,3,256,256]
    G_lr = 2e-4
    D_lr = 2e-4
    betas = [0,5,0.999]
    weight_decay = 1e-5
    step_size = 3000
    gamma = 0.97
    shuffle = True
    nuw_workers = 5
    max_iter = 200000

class SingleCelebADataset(Dataset):
    return 

class MultiCelebADataset(object):
    return 

def test():
    return 

if __name__ == "__main__":
    return 

