# LGMD Vision Model Demonstration Demo

[English](#english) | [中文版](#chinese)

---
<a id="english"></a>
## 📄 Introduction
This project is a video processing demonstration demo based on the biological vision system — the **Lobula Giant Movement Detector (LGMD)**. It supports evaluating both traditional LGMD models and the advanced `BiSTS_LGMD` model for real-time collision (looming) motion detection in video streams.

## ✨ Key Features
* **Multi-Model Support**: Easily switch between classic LGMD variants and the new `BiSTS_LGMD` model.
* **Hardware Acceleration**: Automatically detects and utilizes CUDA for GPU-accelerated inference, with a seamless fallback to CPU.
* **Interactive Visualization**: Real-time rendering using Matplotlib with keyboard shortcut controls for playback status.
* **GUI File Selector**: Automatically pops up a GUI window to browse and select test videos if no explicit file path is provided.

## 📦 Prerequisites
Ensure you have the following core libraries installed in your Python environment:
```bash
pip install opencv-python numpy matplotlib torch
```

---
<a id="chinese"></a>


### 📄 简介
本项目是一个基于生物视觉系统——蝗虫小叶巨型运动检测器（LGMD, Lobula Giant Movement Detector）的视频处理演示 Demo。该 Demo 支持测试传统的 LGMD 模型以及先进的 `BiSTS_LGMD` 模型，用于实时检测视频中的碰撞（隐约接近/Looming）运动。

### ✨ 功能特点
* **多模型支持**：支持轻松切换经典的 LGMD 模型及全新的 `BiSTS_LGMD` 深度学习模型。
* **硬件加速**：自动检测系统环境，并利用 CUDA 进行 GPU 加速推理（若不可用则自动切换为 CPU）。
* **交互式可视化**：使用 Matplotlib 实现实时结果渲染，并支持键盘交互控制播放状态。
* **图形界面选片**：若未指定特定的视频路径，程序会自动弹出 GUI 窗口供用户选择测试视频。

### 📦 环境依赖
请确保你的 Python 环境中已安装以下核心库：
```bash
pip install opencv-python numpy matplotlib torch
```