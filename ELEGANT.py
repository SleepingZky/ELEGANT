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
    def __init__(self):
        break   
    
    # 这个是用来把参数从文件读入
    # input 
    #   -文件地址
    # output
    #   -暂时想到的是加载好参数的模型
    def restore_from_file(self):
        break
    
    # 这个是用来指定模型布局在哪个设备上的
    # input
    #   - model and device
    # output
    #   - None 
    def set_mode_and_gpu(self):
        break
    
    # 不知道这个是啥 还需要研究
    def tensor2var(self, tensors, volatile):
        break
    
    # 不知道这个是啥 还需要研究
    def get_attr_chs(self,encodings,attribute_id):
        break
    
    # 这个应该是 generator 用来生成中间图片的
    # input 
    #   -x 
    # output
    #   -y
    def foward_G(self):
        break

    # 这个是鉴别器 用来判断是不是真样本
    # input
    #   -x1,y1
    # output
    #   -概率
    def forward_D_real_sample(self):
        break

    # 这个也是鉴别器，用来判断是不是假样本
    def forward_D_fake_sample(self,detach):
        break

    # 这个是为 鉴别器 来计算 loss 
    def compute_loss_D(self):
        break

    # 这个是为 生成器 用来计算 loss 
    def compute_loss_D(self):
        break
    
    # 这个应该是用中间结果 再来生成尽可能接近的初始结果
    def backward_D(self):
        break

    # 这个应该是用中间结果 再来生成尽可能接近的初始结果
    def backward_G(self):
        break
    
    # 看名字应该是用来把 归一化了的图片 重新展开成 0-255
    def img_denorm(self,img,scale=255):
        break
    
    # 这个是用来保存训练日志的？
    def save_image_log(self, save_num=20):
        break