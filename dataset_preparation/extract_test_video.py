import os
import random
import base64
from datasets import load_dataset

def main():
    # Load the dataset from Hugging Face using the correct identifier and split
    try:
        # Load the dataset and access the 'train' split
        dataset = load_dataset('ShareGPTVideo/test_raw_video_data', split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check the number of samples in the train split
    total_videos = len(dataset)
    print(f"数据集包含 {total_videos} 个视频样本。")

    # Randomly select 10 samples from the dataset
    num_videos_to_select = min(total_videos, 20)
    selected_indices = random.sample(range(total_videos), num_videos_to_select)

    # Output directory
    output_dir = '../../dataset/extracted_videos'
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save the selected videos
    video_count = 0
    for idx in selected_indices:
        # Extract the video data (assuming the video content is stored in 'mp4' column)
        sample = dataset[idx]

        # Extract the 'mp4' column; change 'mp4' if the column name is different
        video_data = sample.get('mp4')

        # Check if the video data is in bytes format or encoded (e.g., base64)
        if isinstance(video_data, str):
            # Decode the base64 encoded video data
            video_data = base64.b64decode(video_data)

        if isinstance(video_data, bytes):
            output_filename = os.path.join(output_dir, f'video_{video_count}.mp4')
            with open(output_filename, 'wb') as f:
                f.write(video_data)
            video_count += 1
            print(f"已保存视频 {video_count} 个：{output_filename}")
        else:
            print(f"样本 {idx} 的视频数据类型未知或无法解析，跳过。")

    print(f"共随机抽取并保存了 {video_count} 个视频。")

if __name__ == '__main__':
    main()
