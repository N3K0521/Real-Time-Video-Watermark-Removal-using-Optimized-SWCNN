from datasets import load_dataset

# 下载数据集到指定目录
#dataset = load_dataset("ShareGPTVideo/test_raw_video_data", cache_dir="D:/Thesis/Thesis_Watermark/dataset")
ds = load_dataset("samp3209/logo-dataset", cache_dir="/dataset")
