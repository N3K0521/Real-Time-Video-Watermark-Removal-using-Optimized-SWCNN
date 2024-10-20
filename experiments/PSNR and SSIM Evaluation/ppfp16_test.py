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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = get_config('configs/config.yaml')

parser = argparse.ArgumentParser(description="watermark removal from video")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modeldir", type=str, default=config['train_model_out_path_SWCNN'],
                    help='path of model files')
parser.add_argument("--net", type=str, default="HN", help='Network used in test')
parser.add_argument("--video_path", type=str, default='../extracted_videos/320p/video_1_320p.mp4', help='Path to input video')
parser.add_argument("--alpha", type=float, default=1.0, help="The opacity of the watermark")
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
    # Build model
    print('Loading model ...\n')
    if opt.net == "HN":
        net = HN()
    else:
        assert False
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # Load model
    model.load_state_dict(torch.load(os.path.join(opt.modeldir, model_name)))
    model.eval()

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 32) * 32)
    frame_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 32) * 32)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save output video
    output_path = "1.0/output_watermark_removed_ppfp16.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Read and normalize image (each frame from video)
        Img = normalize(np.float32(frame))
        Img = np.expand_dims(Img, 0)
        Img = np.transpose(Img, (0, 3, 1, 2))  # Convert to PyTorch tensor shape (batch, channels, height, width)
        Img = torch.Tensor(Img).cuda()  # Send to GPU directly

        # Mixed precision inference
        with torch.no_grad():
            with autocast():
                Out = torch.clamp(model(Img), 0., 1.)

        # Convert tensors to numpy for displaying and saving
        Out_np = Out.cpu().numpy()[0]

        # Reshape and transpose to make it suitable for displaying and saving
        Out_img = np.transpose(Out_np, (1, 2, 0))
        Out_img = (Out_img * 255).astype(np.uint8)  # Convert to 8-bit for saving

        # Write the processed frame to the output video file
        out.write(Out_img)

        # Merge two images side by side for comparison
        combined_image = np.hstack((frame, Out_img))

        # Convert FPS to a string and draw it on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image as a video stream
        cv2.imshow("Watermark Removal Comparison (Left: Original, Right: Clean)", combined_image)

        # Add a delay to mimic video stream effect
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video and close windows
    cap.release()
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: Generate watermarked video and save (dummy implementation if needed)
    import os

    output_path = f'{opt.video_path[:-4]}_watermarked.mp4'
    if not os.path.exists(output_path):
        create_watermarked_video(opt.video_path, output_path, opt.alpha)

    # Step 2: Read and display watermark removal
    display_watermark_removal(output_path)