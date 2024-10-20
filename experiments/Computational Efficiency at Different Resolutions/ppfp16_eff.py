import cv2
import numpy as np
import os
import argparse
import time
from torch.autograd import Variable
from models import HN
from utils import *
import torch
from torch.cuda.amp import autocast  # For mixed precision
from tqdm import tqdm
import psutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = get_config('configs/config.yaml')

parser = argparse.ArgumentParser(description="watermark removal from video")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='path of model files')
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--video_path", type=str, default='../extracted_videos/1080p/video_1_1080p.mp4',
                    help='Path to input video')
parser.add_argument("--alpha", type=float, default=0.5, help="The opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
opt = parser.parse_args()


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


# 第二步：读取并实时展示去水印的视频
def display_watermark_removal(input_path):
    # 加载模型
    print('Loading model ...\n')
    if opt.net == "HN":
        net = HN()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # 加载预训练模型
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model.eval()

    # 打开带水印的视频文件
    cap = cv2.VideoCapture(input_path)

    width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 32) * 32)  # Resize width to half
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 32) * 32)  # Resize height to half
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open watermarked video.")
        return
    prev_time = time.time()  # Start time for FPS calculation
    cpu_usage_list = []
    gpu_usage_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading video.")
            break

        # 读取并标准化图像
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        frame = cv2.resize(frame, (width, height))
        Img = normalize(np.float32(frame))

        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))  # 转换为PyTorch张量的形状
        Img = torch.Tensor(Img).cuda()  # 发送到GPU
        # print(Img.shape)
        # 推理（混合精度运算）
        with torch.no_grad():
            with autocast():
                Out = torch.clamp(model(Img), 0., 1.)

        # 转换为NumPy数组以进行显示
        Out_np = Out.cpu().numpy()[0]
        Out_img = np.transpose(Out_np, (1, 2, 0))
        Out_img = (Out_img * 255).astype(np.uint8)
        combined_image = np.hstack((frame, Out_img))

        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 记录 CPU 和 GPU 占用率
        cpu_usage = psutil.cpu_percent()
        gpu_usage = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(
            0) * 100 if torch.cuda.is_available() else 0
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

    # 计算并打印平均 CPU 和 GPU 占用率
    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
    avg_gpu_usage = sum(gpu_usage_list) / len(gpu_usage_list) if gpu_usage_list else 0
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}% | Average GPU Usage: {avg_gpu_usage:.2f}%")


if __name__ == "__main__":
    # Step 1: 生成带水印的视频并保存
    import os

    output_path = f'{opt.video_path[:-4]}_watermarked.mp4'
    # create_watermarked_video(opt.video_path, output_path)
    if os.path.exists(output_path):
        pass
    else:
        create_watermarked_video(opt.video_path, output_path, opt.alpha)

    # Step 2: 读取并实时展示去水印的视频
    display_watermark_removal(output_path)