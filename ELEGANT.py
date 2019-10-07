#coding=utf-8
########################################
# 本文件是对于 ELEGANT 论文的复现
# 因为 ELEGANT 的实现看起来比较简单一些
# 所以就挑 ELEGANT 入手
########################################

from dataset import config, MultiCelebADataset
from nets import Encoder, Decoder, Discriminator

import os 
import torch
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain


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
        self.D2 = discriminator(self.n_attributes,self.config.nchw[-1]//2)
        
        self.adv_criterion = torch.nn.BCELoss()
        self.recon_criterion = torch.nn.MSELoss()

        self.restore_from_file()
        self.set_mode_and_gpu()
    
    # 这个是用来把参数从文件读入
    # input 
    #   -文件地址
    # output
    #   -加载好参数的模型
    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_enc)
            ckpt_file_dec = os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.resotre))
            assert os.path.exists(ckpt_file_dec)
            if self.gpu:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec), strict=False)
            else:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc, map_location='cpu'),strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec, map_location='cpu'),strict=False)

            if self.mode == 'train':
                ckpt_file_d1 = os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.resotre))
                assert os.path.exists(ckpt_file_d1)
                ckpt_file_d2 = os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.resotre))
                assert os.path.exists(ckpt_file_d2)

                if self.gpu:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2), strict=False)
                else:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1, map_location='cpu'), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2, map_location='cpu'), strict=False)
    
            self.start_step = self.resotre + 1
        else:
            self.start_step = 1

    # 这个是用来指定模型布局在哪个设备上的以及模式
    # input
    #   - model and device
    # output
    #   - None 
    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.Enc.train()
            self.Dec.trian()
            self.D1.train()
            self.D2.train()
            
            self.writer = SummaryWriter(self.config.log_dir)
            # chain 的用处？
            self.optimizer_G = torch.optim.Adam(chain(self.Enc.parameters(), self.Dec.parameters()),
                                            lr = self.config.G_lr, betas=(0.5,0.999),
                                            weight_decay=self.config.weight_decay)

            self.optimizer_D = torch.optim.Adam(chain(self.D1.parameters(), self.D2.parameters()),
                                            lr=self.config.D_lr, betas=(0.5,0.999),
                                            weight_decay=self.config.weight_decay)

            self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.step_size, gamma=self.config.gamma)
            self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.step_size, gamma=self.config.gamma)
            if self.restore is not None:
                for _ in range(self.resotre):
                    self.G_lr_scheduler.step()
                    self.D_lr_scheduler.step()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()
                    self.D1.cuda()
                    self.D2.cuda()
                    self.adv_criterion.cuda()
                    self.recon_criterion.cuda()
            
            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))
                self.D1 = torch.nn.DataParallel(self.D1, device_ids=list(range(len(self.gpu))))
                self.D2 = torch.nn.DataParallel(self.D2, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'test':
            self.Enc.eval()
            self.Dec.eval()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()
            
            if len(self.gpu)>1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))
        
        else:
            raise NotImplementedError()
    
    # 不知道这个是啥 
    # tensor 和 variable 的关系？？
    def tensor2var(self, tensors, volatile):
        # 为什么要有这一步？
        # 来判断 这个tensors 是否是可迭代的？
        if not hasattr(tensors, '__iter__'):
            tensors = [tensors]
        out=[]
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            # volatile 用来抑制反向传导导数， 但0.4之后已经被移除了？
            var = torch.autograd.Variable(tensor,volatile=volatile)
            out.append(var)
        if len(out) == 1:
            return out[0]
        else:
            return out

    # 不知道这个是啥 还需要研究
    def get_attr_chs(self,encodings,attribute_id):
        num_chs = encodings.size(1)
        # 认为每个特征平均的分配几个信息位
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * attribute_id))
        end = int(np.rint(per_chs * (attribute_id + 1)))
        # return encodings[:,start:end]
        return encodings.narrow(1,start,end-start)
    
    # 这个应该是 generator 用来生成中间图片的
    # input 
    #   -x 
    # output
    #   -y
    def foward_G(self):
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i!=self.attribute_id else self.get_attr_chs(self.z_B, i) for i in range(self.n_attributes)], 1)
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i!=self.attribute_id else self.get_attr_chs(self.z_A, i) for i in range(self.n_attributes)], 1)

        self.R_A = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
        self.R_B = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
        self.R_C = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
        self.R_D = self.Dec(self.z_D, self.z_B, skip=self.B_skip)

        # 为什么这里要用 clamp 到 -1，1呢
        # 明明 Dec 出来的东西（-2，2）之间的
        self.A1 = torch.clamp(self.A + self.R_A, -1, 1)
        self.B1 = torch.clamp(self.B + self.R_B, -1, 1)
        self.C  = torch.clamp(self.A + self.R_C, -1, 1)
        self.D  = torch.clamp(self.B + self.R_D, -1, 1)

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
        self.loss_G.backward()
        self.optimizer_G.step()

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
        for self.step in range(self.start_step, 1+ self.config.max_iter):
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for self.attribute_id in range(self.n_attributes):
                A, y_A = next(self.dataset.gen(self.attribute_id, True))
                B, y_B = next(self.dataset.gen(self.attribute_id, False))
                self.A, self.y_A, self.B, self.y_B = self.tensor2var([A,y_A,B,y_B],volatile=False)

                # forward
                self.foward_G()


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