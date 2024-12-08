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
    twitch_url = "https://video-weaver.lax03.hls.ttvnw.net/v1/playlist/CpQG1RDcDb-ammMKHWSJNi7GMJXz6KPif5A0lLz4xbFbIiEQZAi-o91fmeGDe_O7p1CM5WnCoclDhJhMEm-z5jIEZljrJmIZCmZ3u3Yz0YboPHs3u7rZHHA5GJ1vopgeHNJ4B8FX959inGKGFZ2LmZMh1EuunTo6q6e0q3MCLawgLICv0D5FQaL4KQo_ZsjnJr9QMtSfuFAdtmlwEPHMWCDzd4PdNcOR1GEVi8lgtWTc9G3-_9QPwSPTtxJ6Iju8by0Ludn2gGznFsGeC3Yo-MDXGcsynAS2JujPCRCalVlGcKJui-MjXz-D6Ob0XM6YoTqlhS3Nrh_Day2crYfqare-Lzox1iz22orSUTl9o_IarCZKFkd3EiWkGd7uKAwgLmBqOCsHNV83wOLPgoZeblavIAh5CT49JpOJG_JDOj7dWccS1C8w9ynsz6kHVIFnIEgOQf8WY2pM8tDQA0sqHkWMjfUg33HytCffWBTDqkK90CkajAlt-R47tYOE5k0JLFjnBRexUx0yc_Yth--c97glK8jp4Ue-N8VytFnEVA8a4YEAgjI_Dk-RiAGaf0F6HqLTEeLHdZrxxLdmwMWfDp6nLRCH_Hj5y17Uun_IaGRBXle9D5E9mgt1b_5BDRYachCuZxFWosiGt_0MEevOTcFpJUriOQ1pMYaVhjwVKQD0IkcwekXZaqIDDrenQT8AoZZXgWfUDTEZgYvC5Se5_yuJzMZUT0iU23PMh_N4vP5EaRViX--mDRHSlOL5CWs55YlyDRkAccwGf-Dv0vdVR06kMwO5uGcPekF-Mbe-ZwdsKQ1OIU1DzxO69Azki6U4kvJ2HUd58f5BCYFDJhfWiBt2V_QucD2uxP7xGb4wuTuTBQHrTHLmydHy6WzGTz-7BTmICtEZsADu78XZhTxdVMDBIVneVq10bDWQysnIV3hMvleTnkGq7feaevMDw1KfK_iFGpNruUnEZQBoY5cT2xAryaidtpkFSxEibYTye2AxGOAzSAi4c5E4QH5pDoKZefcCVjiXXie6MBr4UZwA3D-OmeTwWNMaDOhHHc9g0hOxCaPoHiABKgl1cy13ZXN0LTIwhQs.m3u8"  # Replace with the valid Twitch URL
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

        # Convert tensors to NumPy for PSNR calculation
        Out_np = Out.cpu().numpy()[0]
        INoisy_np = INoisy.cpu().numpy()[0]
        Out_img = np.transpose(Out_np, (1, 2, 0))
        INoisy_img = np.transpose(INoisy_np, (1, 2, 0))

        # Calculate PSNR
        psnr = compare_psnr(INoisy_img, Out_img, data_range=1.0)

        # Combine the original and cleaned images for comparison
        combined_image = np.hstack((INoisy_img, Out_img))

        # Display FPS and PSNR on the video
        metrics_text = f"FPS: {fps:.2f} PSNR: {psnr:.2f}"
        cv2.putText(combined_image, metrics_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
