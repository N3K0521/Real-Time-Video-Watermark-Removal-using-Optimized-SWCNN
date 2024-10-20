import os
from datasets import load_dataset
import base64
import requests

def main():
    # 加载数据集
    dataset = load_dataset('ShareGPTVideo/test_raw_video_data')

    # 2. 设置输出目录
    output_dir = '../../dataset/videos'
    os.makedirs(output_dir, exist_ok=True)

    # 3. 初始化计数器
    video_count = 0
    max_videos = 100  # 要提取的视频数量

    # 4. 遍历数据集中的样本
    for idx, sample in enumerate(dataset['train']):  # 使用 'train' 划分
        # 打印样本的字段名，以确定视频数据的字段名
        print(f"样本 {idx} 的字段名：{sample.keys()}")

        # 尝试获取 'mp4' 字段的数据
        video_data = sample.get('mp4')

        if not video_data:
            print(f"样本 {idx} 不包含视频数据，跳过。")
            continue

        # 检查 'mp4' 字段的数据类型
        if isinstance(video_data, str):
            video_url = video_data
            print(f"样本 {idx} 的视频 URL：{video_url}")

            # 使用 requests 下载视频
            try:
                response = requests.get(video_url)
                response.raise_for_status()
                video_content = response.content
            except Exception as e:
                print(f"样本 {idx} 的视频下载失败，错误：{e}")
                continue
        elif isinstance(video_data, bytes):
            # 如果字节数据，直接使用
            video_content = video_data
        else:
            print(f"样本 {idx} 的视频数据类型未知，跳过。")
            continue

        # 保存视频文件
        output_filename = os.path.join(output_dir, f'video_{video_count}.mp4')
        with open(output_filename, 'wb') as f:
            f.write(video_content)

        video_count += 1
        print(f"已保存视频 {video_count} 个。")

        if video_count >= max_videos:
            print(f"已提取 {max_videos} 个视频，程序结束。")
            break

    print(f"共提取了 {video_count} 个视频。")

if __name__ == '__main__':
    main()
