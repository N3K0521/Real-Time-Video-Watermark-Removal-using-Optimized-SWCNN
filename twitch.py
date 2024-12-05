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
    twitch_url = "https://video-weaver.lax03.hls.ttvnw.net/v1/playlist/CqkGFt35S10YwnIN6-xiYdVGWc_swbdEzuImEgyQ4J-e7w6jrf9b-56-Nu15zrPaK_HOXa1EJzZ0zH_Dmt5HXR5cicLHmA6LghvRfIqDf02DbqGPU2TAQLLS7ckWpB53r2rZ9FUB3h-ndm-j68IKqDXE9Kvi9AuogeF6F1TBFXL7RpHMEi6jI7GTCxsZBOiJTZTv_4cjvcw0LoxRiP_egr1ZFwg28JXS47jXq-5riPMNewTPGBRU6CQ_wy-ihn5CFpVZe5i60GQtI3lDbf809qNbO_QfSQF3gCMl3LDcYrNR-mP56p5qtOKbIEyEAXNEdEZF7NYDnysTCBSvpvMqe-iXz5KpA80z7XymyOpV8EweSb5iwK6DkOPWXsxtdycqmx0Van6Pij-1AqXAqixuWYwYdr7F7L2P6OYm83l6iWoFf_Sy2q27hSbR2mTcYdTp69-zKKCCy8jec_7lwkLQ9AsCW3KTOPKQDF4q3Wz6JQJxboPxFddxBoMeRQKd2FXBGrFPA1ap-paAIzaiVKNYId-75fuWwzA2yp9IjNX3qe64mHbUeML89KUFJmewXHVUJyPX2so74CjkyrDNg73OokVyJc_KgEB2geARTZJJI4kwsVO6b-mwKhWvb4J1EhVcsI0diUwyGGsBNiE5rTN2kW14yoIa34XZZhhRZGv5-H9J65yMTX6PUA3UmqswRqlF5qXKLlOv1euCDV33kqYQSygaphDp-mOwVKAZkZ4kdeuhMIyhJSVTQLX4N7q07t4jsQDOC3VGxB5VTlHzwbYJssnnoYAbGRwO9WFWZBzqyIQVpx2KN7uvBE-v_7Lg5ggCJj9_RpaZXHahd5jsOPB3sQjGKfSRgkoB9dGU1bckIef3AE6BMeuk1lKuCaYvDIRW_DeXC9Bo3pjyPgaQD7erheAZF_sTrEeNczzcNkO4bos-mFGVZtCtFsV6L6zWrv0Jt45hw9zpTssYH8F36r4y2bKOpooxjiG20YJa_aLcNvBDj6IEEFxUX24_UPD2OB7HELr_poEzsp5S9gYw4gQilKhvcaHb449PLI4r2wZR7EIVxLbaAYDO0v8jbIIaDFBdRZdHuINa4Kux_iABKgl1cy13ZXN0LTIwgQs.m3u8"  # Replace with the valid Twitch URL
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
        INoisy_np = INoisy.cpu().numpy()[0]
        Out_img = np.transpose(Out_np, (1, 2, 0))
        INoisy_img = np.transpose(INoisy_np, (1, 2, 0))

        # Combine the original and cleaned images for comparison
        combined_image = np.hstack((INoisy_img, Out_img))

        # Display FPS on the video
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the processed video
        cv2.imshow("Twitch Live Stream Watermark Removal (Left: Noisy, Right: Clean)", combined_image)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources and close display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    water_test_twitch_stream()
