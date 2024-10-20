import os
import subprocess

def calculate_metrics(original_video, processed_video):
    # Calculate PSNR
    psnr_command = [
        'ffmpeg', '-i', original_video, '-i', processed_video,
        '-lavfi', 'psnr', '-f', 'null', '-'
    ]
    result = subprocess.run(psnr_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if PSNR calculation was successful
    if result.returncode != 0:
        print(f"Error calculating PSNR for {processed_video}: {result.stderr}")
        return None, None

    # Read PSNR value from stderr
    psnr_value = None
    for line in result.stderr.splitlines():
        if 'average' in line:
            psnr_value = float(line.split('average:')[1].split()[0])
            break

    # Calculate SSIM
    ssim_command = [
        'ffmpeg', '-i', original_video, '-i', processed_video,
        '-lavfi', 'ssim=stats_file=ssim.log', '-f', 'null', '-'
    ]
    result = subprocess.run(ssim_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if SSIM calculation was successful
    if result.returncode != 0:
        print(f"Error calculating SSIM for {processed_video}: {result.stderr}")
        return psnr_value, None

    # Read SSIM value from log file
    ssim_value = None
    try:
        with open('ssim.log', 'r') as log_file:
            log_content = log_file.read()
            print("SSIM log content:", log_content)  # Optional: Print SSIM log for debugging
            for line in log_content.splitlines():
                if 'All:' in line:
                    ssim_value = float(line.split('All:')[1].split()[0])
                    break
    except FileNotFoundError:
        print(f"SSIM log file not found for {processed_video}")

    return psnr_value, ssim_value

# 视频路径
original_video_path = os.path.abspath('../extracted_videos/320p/video_1_320p.mp4')
processed_videos = [
    'output_watermark_removed_original.mp4',
]

# 计算每个处理后视频的 PSNR 和 SSIM
for video in processed_videos:
    psnr, ssim = calculate_metrics(original_video_path, video)
    print(f'Video: {video}')
    print(f'PSNR: {psnr}')
    print(f'SSIM: {ssim}\n')
