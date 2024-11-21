import cv2
import numpy as np
import os
import argparse
import time
from torch.autograd import Variable
from models import HN
from utils import *
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = get_config('configs/config.yaml')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Remove watermark from video")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="Training configuration file")
parser.add_argument("--num_of_layers", type=int, default=17, help="Total number of layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='Path to the model file')
parser.add_argument("--net", type=str, default="HN", help='Network used for testing')
parser.add_argument("--device_index", type=int, default=0, help='Camera device index (default: 0)')
parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='Loss function used during training')
parser.add_argument("--self_supervised", type=str, default="True", help='Enable self-supervised learning')
parser.add_argument("--PN", type=str, default="True", help='Enable perceptual network')
opt = parser.parse_args()

# Configure model naming based on parameters
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

def normalize(data):
    """Normalize input data to range [0, 1]."""
    return data / 255.

def water_test_camera():
    # Build the model
    print('Loading model...\n')
    if opt.net == "HN":
        net = HN()
    else:
        assert False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    # Load the model
    model_path = os.path.join(opt.modeldir, model_name)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Load state_dict and handle "module." prefix issue
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # Remove `module.` prefix
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Open the camera (using DirectShow backend)
    cap = cv2.VideoCapture(opt.device_index, cv2.CAP_DSHOW)  # Enforce DirectShow backend
    if not cap.isOpened():
        print(f"Error: Unable to open the camera (Device Index: {opt.device_index}).")
        return

    # Reduce camera latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = time.time()  # Start time for FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No data captured from the camera.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Read and normalize the image (frame)
        Img = normalize(np.float32(frame))
        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))
        _, _, h, w = Img.shape
        h = int(int(h / 32) * 32)
        w = int(int(w / 32) * 32)
        Img = Img[:, :, :, 0:w]
        Img = Img[:, :, 0:h, :]
        ISource = torch.Tensor(Img)

        # Add watermark noise (adjust or remove this step based on actual needs)
        INoisy = add_watermark_noise_test(ISource, 0., img_id=2, scale_img=1, alpha=opt.alpha)
        INoisy = torch.Tensor(INoisy)
        ISource, INoisy = Variable(ISource.to(device)), Variable(INoisy.to(device))

        # Process noisy frames using the model
        with torch.no_grad():
            Out = torch.clamp(model(INoisy), 0., 1.)

        # Convert tensors to NumPy for display
        Out_np = Out.cpu().numpy()[0]
        INoisy_np = INoisy.cpu().numpy()[0]

        # Reshape and transpose for display
        Out_img = np.transpose(Out_np, (1, 2, 0))
        INoisy_img = np.transpose(INoisy_np, (1, 2, 0))

        # Combine two images side by side for comparison
        combined_image = np.hstack((INoisy_img, Out_img))

        # Convert FPS to string and overlay on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image as a video stream
        cv2.imshow("Watermark Removal Comparison (Left: With Watermark, Right: Without Watermark)", combined_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    water_test_camera()
