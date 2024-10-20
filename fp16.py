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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = get_config('configs/config.yaml')

parser = argparse.ArgumentParser(description="watermark removal from video")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='path of model files')
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--video_path", type=str, default='../extracted_videos/320p/video_1_320p.mp4', help='Path to input video')
parser.add_argument("--alpha", type=float, default=0.5, help="The opacity of the watermark")
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=str, default="True", help='Whether to use perception network')
opt = parser.parse_args()

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
    return data / 255.

def water_test_video():
    # Build model
    print('Loading model ...\n')
    if opt.net == "HN":
        net = HN()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    
    # load model
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model.eval()

    # Open the video file
    cap = cv2.VideoCapture(opt.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Reduce video resolution to speed up processing (optional)
    width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/32)*32)  # Resize width to half
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/32)*32)  # Resize height to half
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev_time = time.time()  # Start time for FPS calculation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading video.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Resize frame (for faster processing)
        frame = cv2.resize(frame, (width, height))

        # Read and normalize image (each frame from video)
        Img = normalize(np.float32(frame))
        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))  # Convert to PyTorch tensor shape (batch, channels, height, width)
        Img = torch.Tensor(Img).cuda()  # Send to GPU directly

        # Add watermark noise (dummy or simulated watermark for this example)
        with torch.no_grad():
            INoisy = add_watermark_noise_test(Img.cpu(), 0., img_id=2, scale_img=1.0, alpha=opt.alpha)
            INoisy = torch.Tensor(INoisy).cuda()

            # Mixed precision inference
            with autocast(dtype=torch.float16):
                Out = torch.clamp(model(INoisy), 0., 1.)

        # Convert tensors to numpy for displaying
        Out_np = Out.cpu().numpy()[0]
        INoisy_np = INoisy.cpu().numpy()[0]

        # Reshape and transpose to make it suitable for displaying
        Out_img = np.transpose(Out_np, (1, 2, 0))
        INoisy_img = np.transpose(INoisy_np, (1, 2, 0))

        # Merge two images side by side for comparison
        combined_image = np.hstack((INoisy_img, Out_img))

        # Convert FPS to a string and draw it on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image as a video stream
        cv2.imshow("Watermark Removal Comparison (Left: Noisy, Right: Clean)", combined_image)

        # Add a delay to mimic video stream effect
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    water_test_video()