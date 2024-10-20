#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-4-11 21:21
# @Author  : 26731
# @File    : flops.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# Modification is made by Huixin Wang
# @Time    : 2024-8-31
# @Author  : Huixin Wang
# @File    : flops.py
# @Software: PyCharm

import torch
import thop
from thop import profile

# from models import UNet, FFDNet, DnCNN, IRCNN, UNet_Atten_4, FastDerainNet, DRDNet, EAFN
# from models import FFDNet, DnCNN, IRCNN, HN, FastDerainNet, DRDNet, EAFN
from models import HN
from torch.autograd import Variable
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
noise_sigma = 0

noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(1)]))
noise_sigma = Variable(noise_sigma)
noise_sigma = noise_sigma.cuda()
# net = UNet_Atten_4()  # 定义好的网络模型
# net = HN()
# net = EAFN()3,48
# net = FastDerainNet(3, 48)
net = HN()  # HN() is the chosen model
net = net.cuda()
input = torch.randn(1, 3, 320, 224)  # 随机生成输入张量，尺寸为(1, 3, 320, 224), 为了适配数据集，非标准输入尺寸可能会影响模型的卷积特性和最终的计算效率
input = input.cuda()
flops, params = profile(net, (input,))
flops, params = thop.clever_format([flops, params], "%.3f")  # 提升结果可读性
print('flops: ', flops, 'params: ', params)
