# BiSTS: A Biphasic Spatial-Temporal Synergy Architecture Improving Separability of Approaching Motion Recognition

[English](#english) | [中文版](#chinese)    [Paper homepage [click here](https://mingshuoxu.github.io/BiSTS-HomePage/))

<a id="english"></a>

BiSTS (biphasic Spatial-Temporal Synergy), is a compact
architecture that jointly exploits short- and long-term spatiotemporal
cues to improve motion separability. This repository contains code and selected data used for developing and evaluating the model.
## Status

This project is currently in pre-publication. Raw code of the proposed BiSTS model has been withheld for review; available components provide reproducible examples and evaluation scripts.
## Repository Structure

The repository is organized as follows (top-level folders and notable files):
- ablation/: Ablation study code and configuration (`ablation.py`, `ablation.json`).
- basic_test/: Scripts to test basic motion patterns (`basic_test.py`, helpers).
- boundary/: Boundary/noise related analysis (`noise_boundary.py`, `noise_boundary.json`).
- demo/: Example demos and usage (`demo.py`, demo readme).
- effectiveness/: Tools to create and evaluate contrast/noise datasets and results (`effe_contrast.py`, `effe_noise.py`).
- lgmd_benchmark/: Benchmarking and evaluation scripts for LGMD models.
- metric_related/: Metric and separability analysis utilities (`Separability.py`).
- model/: Model implementations and modules (LGMD cores and wrappers).
- parameter_analysis/: Parameter search and analysis scripts.
- SWaP_analysis/: Size/Weight/Power analysis scripts.
- unit_test/: Unit tests for model components.
- visulization/: Visualization utilities and plotting scripts.
- `utils.py`: General utility functions used across the repository.
- `readme.md`: This file.
## Quick Start

1. Create a Python virtual environment and install dependencies (see `pyproject.toml`).

2. Run a demo:
    ```bash
    python demo\demo.py
    ```


## Notes
- If you need specific datasets or components that are not present, contact the authors (see below).

## Contact

For questions, collaboration, or data requests, please contact:
- Zheng Yi: zhengyi_06@163.com
- Mingshuo Xu: mingshuoxu@hotmail.com

Thank you for your interest in BiSTS.

---

<a id="chinese"></a>



BiSTS（双相时空协同）是一个紧凑的模型架构，通过联合利用短期和长期的时空线索来提高运动可分性。本仓库包含了用于开发和评估该模型的核心代码以及精选数据集。

## 📢 项目状态

本项目目前处于**发表前阶段（Pre-publication）**。为了便于同行评审，BiSTS模型源文件暂未公开；当前可用组件提供了完整的可复现示例和评估脚本。

## 📂 仓库结构

项目目录组织如下（包含顶层文件夹及核心文件）：

* **`ablation/`**：消融实验代码及配置文件（`ablation.py`, `ablation.json`）。
* **`basic_test/`**：用于测试基础运动模式的脚本（`basic_test.py` 及辅助工具）。
* **`boundary/`**：边界与噪声相关的分析代码（`noise_boundary.py`, `noise_boundary.json`）。
* **`demo/`**：示例演示及使用说明（`demo.py`，含 Demo 自带的 README）。
* **`effectiveness/`**：用于创建和评估对比度/噪声数据集及结果的工具（`effe_contrast.py`, `effe_noise.py`）。
* **`lgmd_benchmark/`**：针对 LGMD 模型的基准测试与评估脚本。
* **`metric_related/`**：指标与可分性分析工具（`Separability.py`）。
* **`model/`**：模型实现与核心模块（包含 LGMD 核心算法及包装器）。
* **`parameter_analysis/`**：参数搜索与分析脚本。
* **`SWaP_analysis/`**：SWaP（体积、重量、功耗）分析脚本。
* **`unit_test/`**：模型组件的单元测试。
* **`visulization/`**：可视化工具与绘图脚本。
* **`utils.py`**：跨模块调用的通用工具函数。
* **`readme.md`**：当前说明文件。

## 🚀 快速入门

1. **环境配置**：创建 Python 虚拟环境并安装所需依赖（具体依赖请参见 `pyproject.toml`）。

2. **运行演示 Demo**：
   ```bash
   python demo/demo.py
    ```
    (注：在 Windows 系统下请使用反斜杠：python demo\demo.py)

## 💡 注意事项
如果您需要仓库中未提供的特定数据集或核心组件，请直接与作者联系（联系方式见下文）。

## 📬 联系方式
如有任何疑问、合作意向或数据请求，请通过以下邮箱联系作者：

Zheng Yi: zhengyi_06@163.com

Mingshuo Xu: mingshuoxu@hotmail.com

感谢您对 BiSTS 项目的关注！
