import cv2
import numpy as np
import os
import argparse
import time
import torch
from models import HN  # 需要有一个HN模型定义在models模块中
from utils import *
from torch.cuda.amp import autocast
from torch import nn
import threading
from queue import Queue

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 初始化配置和命令行参数
config = get_config('configs/config.yaml')
parser = argparse.ArgumentParser(description="Watermark removal from video")
parser.add_argument('--config', type=str, default='configs/config.yaml', help="Training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'], help='Path of model files')
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--video_path", type=str, default='../extracted_videos/320p/video_1_320p.mp4',
                    help='Path to input video')
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

# 用于同步访问输出队列的锁
queue_lock = threading.Lock()


# 批处理多帧的函数
def process_batch(frames, model, output_queue):
    try:
        # 转换成批处理的4D张量
        batch = np.array([normalize(np.float32(frame)) for frame in frames])
        batch = np.transpose(batch, (0, 3, 1, 2))  # 转换为 NCHW 格式
        batch_tensor = torch.Tensor(batch).cuda()

        with torch.no_grad():
            with autocast():
                out_batch = torch.clamp(model(batch_tensor), 0., 1.)

        out_batch_np = out_batch.cpu().numpy()
        out_images = [(np.transpose(out_batch_np[i], (1, 2, 0)) * 255).astype(np.uint8) for i in
                      range(out_batch_np.shape[0])]

        # 将结果放入队列
        with queue_lock:
            for orig_frame, processed_frame in zip(frames, out_images):
                output_queue.put((orig_frame, processed_frame))
    except Exception as e:
        print(f"Failed to process batch: {e}")


# 处理线程
def frame_processing_thread(input_queue, model, output_queue, batch_size):
    frame_batch = []
    while True:
        frame = input_queue.get()
        if frame is None:  # None is the signal to stop the thread
            break
        frame_batch.append(frame)

        # 如果达到批处理大小，处理批次
        if len(frame_batch) == batch_size:
            process_batch(frame_batch, model, output_queue)
            frame_batch.clear()

    # 处理剩余帧
    if len(frame_batch) > 0:
        process_batch(frame_batch, model, output_queue)


# 主函数定义
def display_watermark_removal(input_path, batch_size=2):
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
    width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 32) * 32)  # Resize width to a multiple of 32
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 32) * 32)  # Resize height to a multiple of 32
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("Error: Could not open watermarked video.")
        return

    input_queue = Queue(maxsize=10)  # 控制输入队列的大小
    output_queue = Queue(maxsize=10)  # 控制输出队列的大小

    # 启动处理线程
    processing_thread = threading.Thread(target=frame_processing_thread,
                                         args=(input_queue, model, output_queue, batch_size))
    processing_thread.start()

    frame_count = 0
    start_time = time.time()
    fps = 0  # 每秒帧数

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧放入输入队列
        input_queue.put(frame)

        # 显示已处理的帧
        if not output_queue.empty():
            orig_frame, processed_frame = output_queue.get()
            combined_image = np.hstack((orig_frame, processed_frame))
            frame_count += 1

            # 每秒计算一次帧数
            if time.time() - start_time >= 1.0:
                fps = frame_count
                frame_count = 0
                start_time = time.time()

            # 显示帧数和每秒帧数
            cv2.putText(combined_image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow("Watermark Removal", combined_image)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    # 结束处理线程
    input_queue.put(None)
    processing_thread.join()
    cv2.destroyAllWindows()


# 脚本入口点
if __name__ == "__main__":
    output_path = f'{opt.video_path[:-4]}_watermarked.mp4'
    if not os.path.exists(output_path):
        # 假设有一个函数create_watermarked_video来创建带水印的视频
        print("Generating watermarked video...")
        create_watermarked_video(opt.video_path, output_path, opt.alpha)
    display_watermark_removal(output_path, batch_size=2)  # 设定批处理大小

    # 添加自动关闭窗口
    cv2.destroyAllWindows()