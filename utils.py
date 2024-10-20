import math
import string

import torch
import torch.nn as nn
import numpy as np
import cv2
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize(data):
    return data / 255.
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    Img = np.transpose(Img, (0, 2, 3, 1))
    Iclean = np.transpose(Iclean, (0, 2, 3, 1))
    # print(Iclean.shape)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range,
                             multichannel=True)
    return (SSIM / Img.shape[0])


def batch_RMSE(img, imclean, data_range):
    img = img * 255
    imclean = imclean * 255
    Img = img.data.cpu().numpy().astype(np.uint8)

    Iclean = imclean.data.cpu().numpy().astype(np.uint8)
    MSE = 0
    for i in range(Img.shape[0]):
        MSE += math.sqrt(compare_mse(Iclean[i, :, :, :], Img[i, :, :, :]))
    return (MSE / Img.shape[0])


def add_watermark_noise(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    random_img = random.randint(1, 16) # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "watermark/logo/"
    watermark = Image.open(data_path +'logo'+ str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]

                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.5, 1.0)
            # scale = 1.5
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train


def add_watermark_noise_B(img_train, occupancy=50, self_surpervision=False, same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    random_img = 3  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "../Archived/translucence/"
    watermark = Image.open(data_path + str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    alpha = 0.3 + random.randint(0, 70) * 0.01
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            scale = np.random.uniform(0.5, 1.0)
            # scale = 1.5
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train

# 第一步：生成带水印的视频并保存
def create_watermarked_video(input_path, output_path,alpha):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频属性
    # width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/32)*32)  # Resize width to half
    # height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/32)*32)  # Resize height to half
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/32)*32)  # Resize width to a multiple of 32
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/32)*32)  # Resize height to a multiple of 32
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    with tqdm(total=total_frames, desc="Adding Watermark", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 加水印
            frame = cv2.resize(frame, (width, height))
            Img = normalize(np.float32(frame))
            Img = np.expand_dims(Img, 0)
            Img = np.transpose(Img, (0, 3, 1, 2))  # 转换为PyTorch张量的形状
            Img = torch.Tensor(Img)
            watermarked_frame = add_watermark_noise_test(Img, 0., img_id=2, scale_img=1, alpha=alpha)

            # 转换为NumPy数组
            watermarked_frame_np = np.transpose(watermarked_frame[0], (1, 2, 0))
            watermarked_frame_np = (watermarked_frame_np * 255).astype(np.uint8)

            # 写入带水印的帧
            out.write(watermarked_frame_np)

            # 更新进度条
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Watermarked video saved to {output_path}")
#  这个函数只用来测试
def add_watermark_noise_test(img_train, occupancy=50, img_id=3, scale_img=1.5, self_surpervision=False,
                                same_random=0, alpha=0.3):
    # 加载水印,水印应该是随机加入
    # random_img = random.randint(1, 13)
    # 对比实验的时候选取某个水印进行去除
    random_img = img_id  # "test"  # random.randint(1, 173)
    # Noise2Noise要确保类标和输入的水印为同一张
    if self_surpervision:
        random_img = same_random
    data_path = "./watermark/logo/"
    watermark = Image.open(data_path +'logo'+ str(random_img) + ".png")
    watermark = watermark.convert("RGBA")
    w, h = watermark.size
    # 设置水印透明度
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)  # random.randint(100)
                # color = color[::-1]
                color = color[:-1] + (transparence,)
            watermark.putpixel((i, k), color)
    # watermark = watermark.convert("RGB")
    watermark_np = np.array(watermark)
    watermark_np = watermark_np[:, :, 0:3]
    img_train = img_train.numpy()
    # img_train = Image.fromarray(img_train)
    imgn_train = img_train
    # 数据归一化
    _, water_h, water_w = watermark_np.shape
    occupancy = np.random.uniform(0, occupancy)

    _, _, img_h, img_w = img_train.shape
    # 加载计算占有率的数组
    img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
    # 转成PIL
    img_for_cnt = Image.fromarray(img_for_cnt)
    new_w, new_h = watermark.size
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        tmp = tmp.convert("RGBA")
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        # 转成PIL
        img_for_cnt = Image.fromarray(img_for_cnt)
        while True:
            # 随机选取放缩比例和旋转角度
            angle = random.randint(-45, 45)
            # scale = np.random.uniform(0.5, 1.0)
            # scale = np.random.uniform(7, 8)
            scale = scale_img
            # 旋转水印
            # img = watermark.rotate(angle, expand=1)
            #  放缩水印
            water = watermark.resize((int(w * scale), int(h * scale)))
            # 将噪声转换为PIL
            layer = Image.new("RGBA", tmp.size, (0, 0, 0, 0))
            # 随机选取要粘贴的部位
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(h * scale))  # int(-h * scale)
            x = 128
            y = 128
            # 合并水印文件
            layer.paste(water, (x, y))
            tmp = Image.composite(layer, tmp, layer)

            img_for_cnt.paste(water, (x, y), water)
            img_for_cnt = img_for_cnt.convert("L")
            img_cnt = np.array(img_for_cnt)
            sum = (img_cnt > 0).sum()
            ratio = img_w * img_h * occupancy / 100
            if sum > ratio:
                img_rgb = np.array(tmp).astype(np.float32) / 255.
                img_train[i] = img_rgb[:, :, [0, 1, 2]]
                break
    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return img_train


import torchvision.models as models
from models import VGG16


def load_froze_vgg16():
    # finetunning
    model_pretrain_vgg = models.vgg16(pretrained=True)

    # load VGG16
    net_vgg = VGG16()
    model_dict = net_vgg.state_dict()
    pretrained_dict = model_pretrain_vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # load parameters
    net_vgg.load_state_dict(pretrained_dict)

    for child in net_vgg.children():
        for p in child.parameters():
            p.requires_grad = False
    device_ids = [0]

    model_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).cuda()
    return model_vgg

def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
import yaml


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
