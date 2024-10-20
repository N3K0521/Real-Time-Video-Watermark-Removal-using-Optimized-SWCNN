import os
import ffmpeg

# Define the directory containing the videos
input_dir = '../../dataset/extracted_videos'  # Replace with your video directory if different

# Define the output resolutions and corresponding sizes
resolutions = {
    '320p': (480, 320),   # width, height for 320p
    '480p': (854, 480),   # width, height for 480p
    '720p': (1280, 720),  # width, height for 720p
    '1080p': (1920, 1080) # width, height for 1080p
}

# Create output directories for each resolution if not already existing
for res in resolutions:
    os.makedirs(os.path.join(input_dir, res), exist_ok=True)

# Process each video in the input directory
for video_file in os.listdir(input_dir):
    video_path = os.path.join(input_dir, video_file)

    # Skip directories or non-video files
    if not os.path.isfile(video_path):
        continue

    # Get the video filename without extension
    video_name, ext = os.path.splitext(video_file)

    # Create different resolution versions of each video
    for res, size in resolutions.items():
        output_file = os.path.join(input_dir, res, f"{video_name}_{res}.mp4")

        # Use FFmpeg to resize the video
        try:
            ffmpeg.input(video_path).output(output_file, vf=f"scale={size[0]}:{size[1]}").run()
            print(f"Saved {output_file}")
        except Exception as e:
            print(f"Error processing {video_file} at {res}: {e}")

print("All videos have been processed and saved at different resolutions.")
