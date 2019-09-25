#coding=utf-8
########################################
# 本文件是对于 ELEGANT 论文的复现
# 因为 ELEGANT 的实现看起来比较简单一些
# 所以就挑 ELEGANT 入手
########################################

import os 
import torch
import argparse

import numpy as np
from tensorboardX import SummaryWriter


# 一般就先这样定义一个整体网络的框架的类
class ELEGANT(object):
    # 用来初始化所有的设置, 
    #   -网络部件
    #   -网络参数
    def __init__(self, args, config=config, dataset=MultiCelebADataset, \
                 encoder=Encoder, decoder=Decoder, discriminator=discriminator):
        
        # 基本参数
        self.args = args
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.gpu = args.gpu
        self.mode = args.mode
        self.resotre = agrs.resotre

        # init dataset and networks
        self.config = config
        self.dataset = dataset(self.attributes)
        self.Enc = encoder()
        self.Dec = decoder()
        self.D1 = discriminator(self.n_attributes,self.config.nchw[-1])
        return   
    
    # 这个是用来把参数从文件读入
    # input 
    #   -文件地址
    # output
    #   -暂时想到的是加载好参数的模型
    def restore_from_file(self):
        return
    
    # 这个是用来指定模型布局在哪个设备上的
    # input
    #   - model and device
    # output
    #   - None 
    def set_mode_and_gpu(self):
        return
    
    # 不知道这个是啥 还需要研究
    def tensor2var(self, tensors, volatile):
        return
    
    # 不知道这个是啥 还需要研究
    def get_attr_chs(self,encodings,attribute_id):
        return
    
    # 这个应该是 generator 用来生成中间图片的
    # input 
    #   -x 
    # output
    #   -y
    def foward_G(self):
        return

    # 这个是鉴别器 用来判断是不是真样本
    # input
    #   -x1,y1
    # output
    #   -概率
    def forward_D_real_sample(self):
        return

    # 这个也是鉴别器，用来判断是不是假样本
    def forward_D_fake_sample(self,detach):
        return

    # 这个是为 鉴别器 来计算 loss 
    def compute_loss_D(self):
        return

    # 这个是为 生成器 用来计算 loss 
    def compute_loss_D(self):
        return
    
    # 这个应该是用中间结果 再来生成尽可能接近的初始结果
    def backward_D(self):
        return

    # 这个应该是用中间结果 再来生成尽可能接近的初始结果
    def backward_G(self):
        return
    
    # 看名字应该是用来把 归一化了的图片 重新展开成 0-255
    def img_denorm(self,img,scale=255):
        return
    
    # 这个是用来保存训练日志的？
    def save_image_log(self, save_num=20):
        return

    # 这个应该是用来保存
    def save_sample_images(self, save_num=5):
        return

    # 保存一些标量
    def save_salar_log(self):
        return

    # 保存模型参数
    def save_model(self):
        return

    # 训练的过程
    def train(self):
        return

    # 对图片进行一些预处理
    def transform(self,*images):
        return

    # 对特征进行交换
    def swap(self):
        return

    # 线性调整两者的混合的比例
    def linear(self):
        return

    # 用途不明
    def matrix1(self):
        return

    # 用途不明
    def matrix2(self):
        return

# 调用上述所有的函数
def main():
    return

if __name__ == "__main__":
    main()