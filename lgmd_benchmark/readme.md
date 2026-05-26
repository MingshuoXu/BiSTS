# LGMD Benchmark Custom Model Evaluation

[English](#english) | [简体中文](#chinese)

---

<a id="english"></a>


This project provides an out-of-the-box, plug-and-play evaluation framework. You can use the `eval_custom_model.py` script to easily test and quantify the performance of your newly designed LGMD (Lobula Giant Movement Detector) models on the LGMD benchmark dataset.

### Prerequisites

Before running the evaluation script, ensure you have the following dependencies installed in your Python environment:

* `torch`
* `opencv-python` (cv2)
* `tqdm`
* `prettytable`
* `numpy`

### Quick Start Guide

**Step 1: Prepare the Dataset**
* Ensure you have downloaded the LGMD benchmark dataset folder (containing `.mp4` or `.avi` videos for various scenarios).
* Ensure the ground truth file `annotations.json` is located in the root directory of this dataset folder.

**Step 2: Plug in Your Custom Model**
* Open `eval_custom_model.py` using a text editor or IDE.
* Locate **Step 1** in the code (the `MyCustomLGMDModel` example class).
* Replace this example class with your own model's code.
* Ensure your core inference function meets the following input/output requirements:
  > **Input**: A grayscale image Tensor of shape `[1, 1, H, W]` (already placed on the corresponding CPU/GPU device).
  > **Output**: A scalar float representing the collision/response probability (must between `0.5` and `1.0`).

**Step 3: Run the Benchmark**
* Run the following command in your terminal to start the inference and evaluation pipeline.
* Please replace the path with the actual path to your local dataset.

```bash
python eval_custom_model.py --dataset "/path/to/lgmd_benchmark" --model_name "MySuperLGMD"
```

---

<a id="chinese"></a>
## LGMD 基准测试自定义模型评估

本项目提供了一个开箱即用、即插即用的评估框架。你可以使用 `eval_custom_model.py` 脚本，在 LGMD 基准测试数据集上轻松测试并量化你自己设计的新型 LGMD（Lobula Giant Movement Detector）模型的性能。

### 环境依赖

在运行评估脚本之前，请确保你的 Python 环境中已安装以下依赖库：

* `torch`
* `opencv-python` (cv2)
* `tqdm`
* `prettytable`
* `numpy`

### 快速上手指南

**第 1 步：准备数据集**
* 确保你已经下载了 LGMD 基准测试数据集文件夹（包含不同场景的 `.mp4` 或 `.avi` 视频）。
* 确保该数据集文件夹的根目录下包含真值文件 `annotations.json`。

**第 2 步：接入你的自定义模型**
* 使用文本编辑器或 IDE 打开 `eval_custom_model.py`。
* 找到代码中的 **步骤 1**（`MyCustomLGMDModel` 示例类）。
* 将该示例类替换为你自己的模型代码。
* 确保你的核心推理函数满足以下输入输出条件：
  > **输入**: 形状为 `[1, 1, H, W]` 的灰度图像 Tensor（已放置在相应的 CPU/GPU 设备上）。
  > **输出**: 一个代表碰撞/响应概率的标量浮点数（必须介于 `0.5` 到 `1.0` 之间）。

**第 3 步：运行基准测试**
* 在终端中运行以下命令以启动推理和评估流程。
* 请将路径替换为你本地真实的数据集路径。

```bash
python eval_custom_model.py --dataset "/path/to/lgmd_benchmark" --model_name "MySuperLGMD"
```