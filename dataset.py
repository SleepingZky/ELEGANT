#coding=utf-8
########################################
# 本文件是对于 ELEGANT 论文的复现
# 因为 ELEGANT 的实现看起来比较简单一些
# 所以就挑 ELEGANT 入手
########################################

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
import os,time 
from PIL import Image


# 该文件主要是用来进行基础的配置
class Config:
    # property 是用来让这个函数可以像变量一样被使用
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
        model_dir = os.path.join(self.exp_dir,'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir,'model')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def img_dir(self):
        img_dir = os.path.join(self.exp_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        return img_dir

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

config = Config()


class SingleCelebADataset(Dataset):
    def __init__(self,im_names,labels,config):
        self.im_names = im_names
        self.labels = labels
        self.config = config

    # 每个自己实现的类 都要实现 三个 功能
    # __len__
    # __getitem__
    # __init__
    
    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        image = Image.open(self.im_names[idx])
        image = self.transform(image) * 2 -1
        label = (self.labels[idx] + 1) / 2
        return image, label
    
    @property
    def transform(self):
        # torchvision 里的 transforms 用来做一些图片预处理
        # tramsforms.Compose() 可以用来把一些预处理打包
        transform = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        return transform

    def gen(self):
        # DataLoader 用来生成一个 数据的更迭器
        dataloader = DataLoader(self, batch_size=self.config.nchw[0], shuffle=self.config.shuffle, num_workers=self.config.num_workers, drop_last=True)
        while True:
            for data in dataloader:
                yield data


class MultiCelebADataset(object):
    def __init__(self,attributes,config=config):
        self.attributes = attributes
        self.config = config

        with open(os.path.join(self.config.data_dir,'list_attr_celeba.txt'), 'r') as f:
            # 一次性把label都读入，然后通过split 每行断开了
            lines = f.read().strip().split('\n')
            col_ids = [lines[1].split().index(attribute) + 1 for attribute in self.attributes]
            self.all_labels = np.array([[int(x.split()[col_id]) for col_id in col_ids] for x in lines[2:]], dtype=np.float32)
            self.im_names = np.array([os.path.join(self.config.data_dir, 'align_5p/{:06d}.jpg'.format(idx+1)) for idx in range(len(self.all_labels))])

        self.dict = {i:{True: None, False: None} for i in range(len(self.attributes))}
        # 为了针对不同的的 attributes  有或者没有进行区分 然后得到对应的数据集
        for attribute_id in range(len(self.attributes)):
            for is_positive in [True, False]:
                idxs = np.where(self.all_labels[:,attribute_id] == (int(is_positive)*2 -1))[0]
                im_names = self.im_names[idxs]
                labels = self.all_labels[idxs]
                data_gen = SingleCelebADataset(im_names, labels, self.config).gen()
                self.dict[attribute_id][is_positive] = data_gen
        
    def gen(self, attribute_id, is_positive):
        data_gen = self.dict[attribute_id][is_positive]
        return data_gen

def test():
    dataset = MultiCelebADataset(['Bangs','Smiling'])

    # cProfile 是用来分析 python 的时间性能的
    # 目标是 各个函数
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        if i % 4 == 0:
            images, labels = next(dataset.gen(0, True))
        elif i % 4 == 1:
            images, labels = next(dataset.gen(0,False))
        elif i % 4 == 2:
            images, labels = next(dataset.gen(1,True))
        elif i % 4 == 3:
            images, labels = next(dataset.gen(1,False))
        print(i)
        print(images.shape, labels.shape)
        print(labels.numpy())

    pr.disable()
    # IPython 调用 交互式 模式
    from IPython import embed; embed(); exit()
    pr.print_stats(sort='tottime')


if __name__ == "__main__":
    data_dir = 'D:/腾讯实习/dataset/CelebA/Anno'
    attributes = ['Bangs','Mustache']
    with open(os.path.join(data_dir,'list_attr_celeba.txt'), 'r') as f:
        lines = f.read().strip().split('\n')
        col_ids = [lines[1].split().index(attribute) + 1 for attribute in attributes]
        all_labels = np.array([[int(x.split()[col_id]) for col_id in col_ids] for x in lines[2:]], dtype=np.float32)
        print(all_labels)

        idxs = np.where(all_labels[:,0] == 1)
        print(idxs)

    test()
