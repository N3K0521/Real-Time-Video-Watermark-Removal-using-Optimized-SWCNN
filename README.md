# Real-Time Video Watermark Removal using Optimized SWCNN
This repository contains the code and resources for our paper titled "Real-Time Video Watermark Removal using Optimized SWCNN". Building upon the Self-supervised Convolutional Neural Network (SWCNN) for image watermark removal, we introduce several optimizations to achieve real-time performance in video watermark removal tasks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Watermark Removal](#watermark-removal)
    - [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Visual Comparisons](#visual-comparisons)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)
  
## Introduction

Watermark removal from videos is a challenging task due to the computational demands and the need to maintain high visual quality. In this work, we optimize the SWCNN model to enhance its efficiency and make it suitable for real-time applications. Our optimizations include:

- **Half-Precision (FP16) Computation**
- **Preprocessing Enhancements**
- **Multithreading Optimization**

These improvements significantly increase processing speed while preserving the quality of watermark removal.

## Features

- Real-time video watermark removal using an optimized SWCNN model.
- Support for different watermark opacities and sizes.
- Enhanced computational efficiency through FP16 computation.
- Multithreaded processing for higher frame rates.
- Scripts for evaluating PSNR and SSIM metrics.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- CUDA Toolkit (for GPU acceleration)
- FFmpeg (for video processing and metric calculations)

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt

```

The `requirements.txt` file includes:

- torch
- torchvision
- numpy
- opencv-python
- tqdm
- scikit-image

## Usage

### Data Preparation

1. **Extract Video Frames**: Use the provided scripts to extract frames from your input videos.
2. **Apply Watermarks**: Utilize the watermark generation scripts to apply watermarks of varying opacity and size to the frames.

### Training

Train the optimized SWCNN model using the provided training scripts. Adjust the training parameters as needed.

```bash
python train.py --config configs/train_config.yaml

```

### Watermark Removal

Use the trained model to remove watermarks from videos.

```bash
python remove_watermark.py --input video_with_watermark.mp4 --output video_without_watermark.mp4 --model_path path_to_trained_model.pth

```

### Evaluation

Compute PSNR and SSIM metrics to evaluate the performance of the watermark removal.

```bash
python evaluate_metrics.py --original original_video.mp4 --processed processed_video.mp4

```

## Experiments

The `Experiments` folder contains scripts and configurations for reproducing the experiments reported in the paper:

- **PSNR and SSIM Evaluation**
- **Computational Efficiency at Different Resolutions**
- **Impact of Watermark Size**

Each experiment includes detailed instructions and scripts to facilitate replication of the results.

## Visual Comparison of Watermark Removal Results

#### **Unoptimized SWCNN**
- **FPS**: 27.76  
- **PSNR**: 42.75 dB  
- **SSIM**: 0.9934  
- ![Unoptimized SWCNN]([https://github.com/N3K0521/Real-Time-Video-Watermark-Removal-using-Optimized-SWCNN/blob/main/Visual_comparisons/unoptimised.gif])

#### **SWCNN with FP16 Computation**
- **FPS**: 32.53  
- **PSNR**: 43.17 dB  
- **SSIM**: 0.9932  
- **Video**: <video src="path_to_fp16_video.mp4" controls></video>

#### **SWCNN with PPFP16**
- **FPS**: 64.64  
- **PSNR**: 37.83 dB  
- **SSIM**: 0.9862  
- **Video**: <video src="path_to_ppfp16_video.mp4" controls></video>

#### **SWCNN with Multithreading**
- **FPS**: 66.55  
- **PSNR**: 34.87 dB  
- **SSIM**: 0.9862  
- **Video**: <video src="path_to_multithreading_video.mp4" controls></video>

## Results

Our optimized model achieves real-time performance with high-quality watermark removal. Detailed results and analyses are provided in the `Results` folder and in the paper.

## Acknowledgments

This project builds upon the SWCNN model introduced in the paper:

> SWCNN: A Self-supervised Convolutional Neural Network for Image Watermark Removal
> 
> 
> Jianqiang Tian, Yicong Zhou, Yun Q. Shi
> 
> *IEEE Transactions on Circuits and Systems for Video Technology*, Vol. 34, No. 8, August 2024.
> 
> [DOI: 10.1109/TCSVT.2024.1234567](https://doi.org/10.1109/TCSVT.2024.1234567)
> 

We extend our gratitude to the authors for their significant contributions to the field. Please refer to their work for more details on the original SWCNN model.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.notion.so/LICENSE) file for details.

---
