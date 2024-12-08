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
from collections import OrderedDict

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load configuration
config = get_config('configs/config.yaml')

# Argument parser for input parameters
parser = argparse.ArgumentParser(description="Watermark removal from Twitch live stream")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="Training configuration file")
parser.add_argument("--num_of_layers", type=int, default=17, help="Total number of layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='Path to the model files')
parser.add_argument("--net", type=str, default="HN", help='Network to be used for testing')
parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='Loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='Whether to use self-supervised learning')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
opt = parser.parse_args()

# Model naming convention
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


# Normalize the input data
def normalize(data):
    return data / 255.


# Function to overlay fake illegal advertisement
def add_fake_overlay(frame):
    # Example text for illegal overlay
    overlay_text = "Join Now: Best Online Casino! Visit www.fakecasino.com"
    # Add text at a specific position with custom styling
    cv2.putText(frame, overlay_text, (50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# Watermark removal from Twitch live stream
def water_test_twitch_stream():
    # Load the model
    print('Loading model...\n')
    if opt.net == "HN":
        net = HN()
    else:
        raise ValueError("Unsupported network type.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    # Load the model weights
    model_path = os.path.join(opt.modeldir, model_name)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
    model.eval()

    # Twitch live stream URL
    twitch_url = "https://video-weaver.lax03.hls.ttvnw.net/v1/playlist/CpMG3TL0aMIZ3utBK8LEqs1SiH8N5nNavnLP4y94aQMpJyQBz_o7RteOt5YJp5qMmsIkUxe0wAqJ8xm3yR3asS9EZFco94z-kjw3aYCF2z1JpjqhHqyCUOtieUZ3vla8o91MiIJJSd6VS46H2XrEkcyntr1BzDW3aywNRUiKozkvehp-WuyyjL5_KWnaO6JswARzV4j998O1esfFkkF1nMK0fC9onqMQEmT0OwdtdLLz-Q1FKIk8DYhQH-fYb-gFYRhXfM_Rj0UbEQWD7K1evtMVKmBOx8AKOFBKKT0zH0s00OBaPuRDvcyp8MWrttEaEJqgyNwogZXZiP8yUvFjVjAzW162xhOHW-ey--Rwhn94QGjjntH5nEVUHudZlmv3NyCBozazY6tps7nP8Micc6lTdvSrX_7EuM36BYmctbUIu7aD0xKOopGBsTO3oRceDj7XvxeWaiMG2_yBxE6Hy1hg9YVtG2PVxvXQ-nIR6k3x5VRq2uco8qsga5f6deGrWHtJqvZe-WDSb-GkZxsUyYFiDmFgtaJCifRriGKDpkUIUt2DGSB7wAlkbt9yieSTqvRULAZGIWc0iid7-rdqxp0e34-1jZDImhNtqaLLaSeOIbD1wMejX31mS2IcjjecOwHJB6Xg04rF6158UAqN0GkWPsDXL-ICq55_79mFMX3k02L5uy4jm2qHC_9MX7W0pCD5BYkdUWurCWNnuVA-wX2zFtJalBcvRYu1_ht6TF6hcmwk0qVeJ6j8KH4aBFRPEFnpVp_p4tT4mMBfoO-0f7795C0NMrOjBJTDL2G1seD7Rg9aErgtWTngi23mf5dec5gIPGbEFzc6MC-fy0xk-nLhRfdv1pYYPzQv_8Peivk8b7nIDqIc9_8ZOlr2t-z3s6MUW2OK4noJEn5mS5ormjVPC-akz4q4TzWlTX4lP4oVw5_j5PmyweI-8Spx63g_22SuMc-qteudxyYzNbG3tkDSXAS46eSh7A7XRrYO9C8k9wfls-BJqb7ix-XsiMTUgwQWFnYtteKyPf72E6Oev3bzkASXhBoM2XLhuCFMRfZltYAqIAEqCXVzLXdlc3QtMjCFCw.m3u8"  # Replace with the valid Twitch URL
    cap = cv2.VideoCapture(twitch_url)

    # Check if the stream is accessible
    if not cap.isOpened():
        print("Could not connect to the Twitch live stream.")
        return

    prev_time = time.time()  # Start time for FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read data from the live stream.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Resize frame to a multiple of 32 (for model compatibility)
        height, width = frame.shape[:2]
        height = (height // 32) * 32
        width = (width // 32) * 32
        frame = cv2.resize(frame, (width, height))

        # Normalize and prepare frame for inference
        Img = normalize(np.float32(frame))
        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))  # Convert to PyTorch tensor shape
        Img = torch.Tensor(Img).to(device)

        # Add simulated watermark noise
        with torch.no_grad():
            INoisy = add_watermark_noise_test(Img.cpu(), 0., img_id=8, scale_img=1.0, alpha=opt.alpha)
            INoisy = torch.Tensor(INoisy).to(device)

            # Use mixed precision for inference
            with autocast(dtype=torch.float16):
                Out = torch.clamp(model(INoisy), 0., 1.)

        # Convert tensors to NumPy for display
        Out_np = Out.cpu().numpy()[0]
        Out_img = np.transpose(Out_np, (1, 2, 0))

        # Add fake overlay to the cleaned image
        Out_img_with_overlay = add_fake_overlay((Out_img * 255).astype(np.uint8))

        # Display FPS on the video
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(Out_img_with_overlay, fps_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the processed video
        cv2.imshow("Processed Stream with Fake Overlay", Out_img_with_overlay)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources and close display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    water_test_twitch_stream()
