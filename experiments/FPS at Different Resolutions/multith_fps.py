import cv2
import numpy as np
import os
import argparse
import time
import torch
from models import HN
from utils import *
from torch.cuda.amp import autocast
from torch import nn
import psutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 初始化配置和命令行参数
config = get_config('configs/config.yaml')
parser = argparse.ArgumentParser(description="Watermark removal from video")
parser.add_argument('--config', type=str, default='configs/config.yaml', help="Training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'], help='Path of model files')
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--video_path", type=str, default='../extracted_videos/720p/video_1_720p.mp4', help='Path to input video')
parser.add_argument("--alpha", type=float, default=0.5, help="The opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='Whether self-supervised')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
opt = parser.parse_args()

# 辅助函数定义
def normalize(data):
    return data / 255.

if opt.PN == "True":
    model_name_1 = "per"
else:
    model_name_1 = "woper"
if opt.loss == "L1":
    model_name_2 = "L1"
else:
    model_name_2 = "L2"
if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"

tensorboard_name = opt.net + model_name_1 + model_name_2 + model_name_3 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"

# 主函数定义
def display_watermark_removal(input_path):
    print('Loading model ...\n')
    if opt.net == "HN":
        net = HN()
    else:
        assert False, "Network type is not supported"
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model.eval()

    cap = cv2.VideoCapture(input_path)
    width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 32) * 32)
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 32) * 32)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open watermarked video.")
        return

    cpu_usage_list = []
    gpu_usage_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Read and normalize image (each frame from video)
        frame = cv2.resize(frame, (width, height))
        Img = normalize(np.float32(frame))
        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))  # Convert to PyTorch tensor shape
        Img = torch.Tensor(Img).cuda()

        # Inference (Mixed Precision)
        with torch.no_grad():
            with autocast():
                Out = torch.clamp(model(Img), 0., 1.)

        # Convert tensors to numpy for displaying
        Out_np = Out.cpu().numpy()[0]
        Out_img = np.transpose(Out_np, (1, 2, 0))
        Out_img = (Out_img * 255).astype(np.uint8)
        combined_image = np.hstack((frame, Out_img))

        # Display CPU and GPU usage
        cpu_usage = psutil.cpu_percent()
        gpu_usage = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(0) * 100 if torch.cuda.is_available() else 0
        cpu_usage_list.append(cpu_usage)
        gpu_usage_list.append(gpu_usage)
        print(f"CPU Usage: {cpu_usage:.2f}% | GPU Usage: {gpu_usage:.2f}%")

        # 实时展示去水印的视频
        cv2.imshow("Watermark Removal", combined_image)

        # 添加延迟以模仿视频流效果
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print average CPU and GPU usage after video is processed
    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
    avg_gpu_usage = sum(gpu_usage_list) / len(gpu_usage_list) if gpu_usage_list else 0
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}% | Average GPU Usage: {avg_gpu_usage:.2f}%")

# 脚本入口点
if __name__ == "__main__":
    output_path = f'{opt.video_path[:-4]}_watermarked.mp4'
    if not os.path.exists(output_path):
        print("Generating watermarked video...")
        create_watermarked_video(opt.video_path, output_path, opt.alpha)
    display_watermark_removal(output_path)