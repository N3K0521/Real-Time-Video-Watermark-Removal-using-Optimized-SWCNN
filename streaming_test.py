import cv2
import numpy as np
import os
import argparse
import time
from torch.autograd import Variable
from models import HN
from utils import *
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn as nn
from torch.cuda.amp import autocast  # Import autocast for mixed precision inference

# Set GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load configuration
config = get_config('configs/config.yaml')

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Remove watermarks from video streams")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="Configuration file for training")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of layers in the model")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='Path to the trained model file')
parser.add_argument("--net", type=str, default="HN", help='Network architecture to use for testing')
parser.add_argument("--device_index", type=int, default=0, help='Camera device index (default: 0)')
parser.add_argument("--alpha", type=float, default=0.5, help="Watermark opacity level")
parser.add_argument("--loss", type=str, default="L1", help='Loss function used during training')
parser.add_argument("--self_supervised", type=str, default="True", help='Enable self-supervised learning')
parser.add_argument("--PN", type=str, default="True", help='Enable perceptual network usage')
opt = parser.parse_args()

# Define model naming conventions based on arguments
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

# Normalize image data
def normalize(data):
    return data / 255.

# Function for removing watermarks using camera input
def water_test_camera():
    print('Loading the model...\n')
    if opt.net == "HN":
        net = HN()
    else:
        raise ValueError("Unsupported network specified")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    # Load the trained model
    model_path = os.path.join(opt.modeldir, model_name)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Open camera
    cap = cv2.VideoCapture(opt.device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Unable to open camera (Device Index: {opt.device_index}).")
        return

    # Reduce camera latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = time.time()

    psnr_list = []  # List to store PSNR values for each frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No data captured from the camera.")
                break

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Preprocess frame
            Img = normalize(np.float32(frame))
            Img = np.expand_dims(Img, 0)
            Img = np.transpose(Img, (0, 3, 1, 2))
            _, _, h, w = Img.shape
            h = int(int(h / 32) * 32)
            w = int(int(w / 32) * 32)
            Img = Img[:, :, 0:h, 0:w]
            ISource = torch.Tensor(Img).to(device)

            # Add watermark noise for testing
            INoisy = add_watermark_noise_test(ISource.cpu(), 0., img_id=8, scale_img=1, alpha=opt.alpha)
            INoisy = torch.Tensor(INoisy).to(device)

            # Inference with mixed precision
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    Out = torch.clamp(model(INoisy), 0., 1.)

            # Convert results for PSNR
            Out_np = Out.cpu().float().numpy()[0]
            INoisy_np = INoisy.cpu().float().numpy()[0]
            Out_img = np.transpose(Out_np, (1, 2, 0))
            INoisy_img = np.transpose(INoisy_np, (1, 2, 0))

            # Calculate PSNR
            psnr = compare_psnr(INoisy_img, Out_img, data_range=1.0)
            psnr_list.append(psnr)

            # Display PSNR in real-time
            metrics_text = f"FPS: {fps:.2f} PSNR: {psnr:.2f}"
            combined_image = np.hstack((INoisy_img, Out_img))
            cv2.putText(combined_image, metrics_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Watermark Removal Comparison (Left: Watermarked, Right: Cleaned)", combined_image)

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStream interrupted by user.")

    finally:
        # Ensure resources are released even in case of an error
        cap.release()
        cv2.destroyAllWindows()

        # Calculate and print average PSNR
        avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
        print(f"Average PSNR: {avg_psnr:.2f}")

if __name__ == "__main__":
    water_test_camera()
