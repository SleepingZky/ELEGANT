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
            # chain 的用处
            # 把model的参数拼在了一起，是一个生成器
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
    # 之前版本中 tensor 和 variable 有区别，0.4之后就没有区别了
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

    # 这个用来得到特征对应的编码的位置
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
        self.d1_A = self.D1(self.A, self.y_A)
        self.d1_B = self.D1(self.B, self.y_B)
        self.d2_A = self.D2(self.A, self.y_A)
        self.d2_B = self.D2(self.A, self.y_B)

    # 这个也是鉴别器，用来判断是不是假样本
    def forward_D_fake_sample(self,detach):
        self.y_C, self.y_D = self.y_A.clone(), self.y_B.clone()
        self.y_C.data[:,self.attribute_id] = self.y_B.data[:,self.attribute_id]
        self.y_D.data[:,self.attribute_id] = self.y_A.data[:,self.attribute_id]

        # detach() 防止再传入Dis的过程中 self.C 被篡改？
        if detach:
            self.d1_C = self.D1(self.C.detach(), self.y_C)
            self.d1_D = self.D1(self.D.detach(), self.y_D)
            self.d2_C = self.D1(self.C.detach(), self.y_C)
            self.d2_D = self.D1(self.D.detach(), self.y_D)
        else:
            self.d1_C = self.D1(self.C, self.y_C)
            self.d1_D = self.D1(self.D, self.y_D)
            self.d2_C = self.D1(self.C, self.y_C)
            self.d2_D = self.D1(self.D, self.y_D)

    # 这个是为 鉴别器 来计算 loss 
    def compute_loss_D(self):
        self.D_loss={
            'D1':   self.adv_criterion(self.d1_A, torch_ones_like(self.d1_A)) + \
                    self.adv_criterion(self.d1_B, torch.ones_like(self.d1_B)) + \
                    self.adv_criterion(self.d1_C, torch.zeros_like(self.d1_C)) + \
                    self.adv_criterion(self.d1_D, torch.zeros_like(self.d1_D)),
            
            'D2':   self.adv_criterion(self.d2_A, torch.ones_like(self.d2_A)) + \
                    self.adv_criterion(self.d2_B, torch.ones_like(self.d2_B)) + \   
                    self.adv_criterion(self.d2_C, torch.zeros_like(self.d2_C)) + \
                    self.adv_criterion(self.d2_D, torch.zeros_like(self.d2_D))
        }
        self.loss_D = (self.D_loss['D1'] + 0.5*self.D_loss['D2']) / 4

    # 这个是为 生成器 用来计算 loss 
    def compute_loss_G(self):
        self.G_loss={
            'reconstruction': self.recon_criterion(self.A1,self.A) + self.recon_criterion(self.B1, self.B),
            'adv1': self.adv_criterion(self.d1_C, torch.ones_like(self.d1_C)) + \
                    self.adv_criterion(self.d1_D, torch.ones_like(self.d1_D)),
            'adv2': self.adv_criterion(self.d2_C, torch.ones_like(self.d2_C)) + \
                    self.adv_criterion(self.d2_D, torch.ones_like(self.d2_D)),
        }
        self.loss_G = 5 * self.G_loss['reconstruction'] + self.G_loss['adv1'] + 0.5 * self.G_loss['adv2']

    # 这个应该是用中间结果 再来生成尽可能接近的初始结果
    def backward_D(self):
        self.loss_D.backward()
        self.optimizer_D.step()

    # 把 loss 回传 然后优化G
    def backward_G(self):
        self.loss_G.backward()
        self.optimizer_G.step()

    # 看名字应该是用来把 归一化了的图片 重新展开成 0-255
    def img_denorm(self,img,scale=255):
        return (img + 1) * scale / 2.
    
    # 这个是用来保存训练日志的？
    def save_image_log(self, save_num=20):
        image_info = {
            'A/img' : self.img_denorm(self.A.data.cpu(), 1)[:save_num],
            'B/img' : self.img_denorm(self.B.data.cpu(), 1)[:save_num],
            'C/img' : self.img_denorm(self.C.data.cpu(), 1)[:save_num],
            'D/img' : self.img_denorm(self.D.data.cpu(), 1)[:save_num],
            'A1/img' : self.img_denorm(self.A1.data.cpu(), 1)[:save_num],
            'B1/img' : self.img_denorm(self.B1.data.cpu(), 1)[:save_num],
            'R_A/img' : self.img_denorm(self.R_A.data.cpu(), 1)[:save_num],
            'R_B/img' : self.img_denorm(self.R_B.data.cpu(), 1)[:save_num],
            'R_C/img' : self.img_denorm(self.R_C.data.cpu(), 1)[:save_num],
            'R_D/img' : self.img_denorm(self.R_D.data.cpu(), 1)[:save_num],
        }
        for tag, images in image_info.items():
            for idx, images in enumerate(images):
                self.writer.add_image(tag+'/{}_{:02d}'.format(self.attribute_id,idx), image, self.step)

    # 这个应该是用来保存
    def save_sample_images(self, save_num=5):
        canvas = torch.cat((self.A, self.B, self.C, self.D, self.A1, self.B1), -1)
        img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()),(0,2,3,1)).astype(np.uint8)
        for i in range(save_num):
            Image.fromarray(img_array[i]).save(os.path.join(self.config.img_dir, 'step_{:06d}_attr_{}_{:02d}.jpg'.format(self.step, self.attribute_id, i)))

    # 保存一些标量
    def save_scalar_log(self):
        scalar_info= {
            'loss_D': self.loss_D.data.cpu().numpy()[0],
            'loss_G': self.loss_G.data.cpu().numpy()[0],
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0]
        }

        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value.item()

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value.item()

        for tag, value in scalar_info.items():
            self.writer.add_scaler(tag, value, self.step)

    # 保存模型参数
    def save_model(self):
        reduced = lambda key: key[7:] if key.startswith('module.') else key
        torch.save({reduced(key): val.cpu() for key, val in self.Enc.state_dict().items()}, os.path.join(self.config.model_dir,'Enc_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.Dec.state_dict().items()}, os.path.join(self.config.model_dir,'Dec_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D1.state_dict().items()}, os.path.join(self.config.model_dir,'Dec_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D2.state_dict().items()}, os.path.join(self.config.model_dir,'Dec_iter_{:06d}.pth'.format(self.step)))

    # 训练的过程
    def train(self):
        for self.step in range(self.start_step, 1+ self.config.max_iter):
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for self.attribute_id in range(self.n_attributes):
                A, y_A = next(self.dataset.gen(self.attribute_id, True))
                B, y_B = next(self.dataset.gen(self.attribute_id, False))
                self.A, self.y_A, self.B, self.y_B = self.tensor2var([A,y_A,B,y_B], volatile=False)

                # forward
                self.foward_G()

                # update D
                self.forward_D_real_sample()
                self.forward_D_fake_sample(detach=True)
                self.compute_loss_D()
                self.optimizer_D.zero_grad()
                self.backward_D()

                # updata G
                self.forward_D_fake_sample(detach=False)
                self.compute_loss_G()
                self.optimizer_G.zero_grad()
                self.backward_G()

                if self.step % 100 == 0:
                    self.save_image_log()

                if self.step % 2000 == 0:
                    self.save_model()
            
            print('step: %06d, loss D: %.6f, loss G: %.6f' % (self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            if self.step % 100 == 0:
                self.save_scalar_log()

            if self.step % 2000 == 0:
                self.save_model()

        print('Finished Training!')
        self.writer.close()

    # 对图片进行一些预处理
    def transform(self,*images):
        transform1 = transform.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1 
        out = [transform2(transform1(image)) for iamge in images]
        return out

    # 对特征进行交换
    # 完全就是两两交换
    def swap(self):
        self.attribute_id = self.args.swap_list[0]
        self.B, self.A = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0])),volatile=True)
        
        self.forward_G()
        img = torch.cat((self.B, self.A, self.D, self.C), -1)
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()),(0,2,3,1)).astype(np.unit8)[0]
        Image.fromnarray(img).save('swap.jpg')

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