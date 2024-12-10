# Real-Time Video Watermark Removal using Optimized SWCNN

This repository contains the code and resources for the paper:  
**"Deep Learning-Based Attacks on Traditional Watermarking Systems in Real-Time Live Video Streams"**

Our work builds upon the Self-Supervised Convolutional Neural Network (SWCNN) for image watermark removal. We introduce optimizations—such as half-precision (FP16) inference, preprocessing enhancements, and multithreaded processing—to enable real-time watermark removal in live streaming scenarios. The methods and experiments described in the paper demonstrate the ability to handle both locally served and online (e.g., Twitch) live video streams.

## Key Features

- **Real-time Watermark Removal**: Extends SWCNN to video inputs, enabling watermark removal at live streaming frame rates.
- **FP16 Optimization**: Employs half-precision computations to improve inference speed without compromising watermark removal quality.
- **Preprocessing and Multithreading**: Integrates preprocessing strategies and multithreaded pipelines to maintain low latency and high FPS.
- **Local and Online Stream Support**: Demonstrates effectiveness on both local streams (via VLC) and online streams (e.g., Twitch HLS).

## Code Overview

The repository provides two primary scripts mentioned in the paper:

1. **`streaming.py`**  
   - Designed for local streaming scenarios.  
   - Captures frames from a locally served video stream (e.g., served by VLC at `http://localhost:8080`) using OpenCV’s `VideoCapture`.  
   - Normalizes, resizes, and converts frames to tensors, then processes them through the SWCNN model with FP16 inference.  
   - Displays original and watermark-removed frames side-by-side in real time.
   
2. **`simulation_real_case.py`**  
   - Targets online streaming from platforms like Twitch.  
   - Accepts an HLS (m3u8) URL, extracted from browser developer tools, to capture live segments.  
   - Handles unstable network conditions, varying bitrates, and changing GOP structures.  
   - Maintains watermark removal quality, adapting to lower resolutions when necessary and employing FP16 computations for efficiency.  
   - Demonstrates how the approach can be used in realistic, potentially adversarial scenarios, including overlaying illicit content after watermark removal.

These scripts reflect the pipeline and experimental setups described in the paper. No additional code beyond what the paper discusses is included here.

## Usage

### Prerequisites

- Python 3.9
- NVIDIA GPU with CUDA support
- Dependencies listed in the project’s `requirements.txt`

### Watermark Removal

Use the trained model to remove watermarks from Local streaming:

Run the local streaming script:
```bash
python streaming.py
```

Run the Twitch streaming script: 
```bash
python twitch.py
```

Run the real case simulation script: 
```bash
python simulation_real_case.py
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
- **Video**: [Unoptimized SWCNN Video](https://raw.githubusercontent.com/N3K0521/Real-Time-Video-Watermark-Removal-using-Optimized-SWCNN/main/Visual_comparisons/fp16.mp4)


#### **SWCNN with FP16 Computation**
- **FPS**: 32.53  
- **PSNR**: 43.17 dB  
- **SSIM**: 0.9932  
- **Video**: <video src="path_to_fp16_video.mp4" controls></video>
- **Video**: [SWCNN with FP16 Computation](https://raw.githubusercontent.com/N3K0521/Real-Time-Video-Watermark-Removal-using-Optimized-SWCNN/main/Visual_comparisons/fp16.mp4)

#### **SWCNN with PPFP16**
- **FPS**: 64.64  
- **PSNR**: 37.83 dB  
- **SSIM**: 0.9862  
- **Video**: [SWCNN with PPFP16](https://raw.githubusercontent.com/N3K0521/Real-Time-Video-Watermark-Removal-using-Optimized-SWCNN/main/Visual_comparisons/ppfp16.mp4)

#### **SWCNN with Multithreading**
- **FPS**: 66.55  
- **PSNR**: 34.87 dB  
- **SSIM**: 0.9862  
- **Video**: [SWCNN with Multithreading](https://raw.githubusercontent.com/N3K0521/Real-Time-Video-Watermark-Removal-using-Optimized-SWCNN/main/Visual_comparisons/mm.mp4)

#### **Local Streaming Demo**

[Watch Local Streaming Demo](https://github.com/your-repo-path/local-streaming-demo.mp4)

#### **Twitch Streaming Demo**

[Watch Twitch Streaming Demo](https://github.com/your-repo-path/twitch-streaming-demo.mp4)

#### **Real Case Simulation Streaming Demo**

[Watch Real Case Simulation Streaming Demo](https://github.com/your-repo-path/real-case-simulation-demo.mp4)


## Results

Our optimized model achieves real-time performance with high-quality watermark removal. Detailed results and analyses are provided in the paper.

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

