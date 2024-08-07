<h1 align="center">
    PacGoc: 基于紫光同创 FPGA 的音频处理系统
</h1>
<div align="center">

<!-- <img src="README.assets/header.svg" width="750"></img> -->

<br>

<p>
<a href="https://github.com/Imiloin/PacGoc"><img alt="Github Repository" src="https://img.shields.io/badge/Github-Repository-green?style=for-the-badge&logo=github&logoColor=green"></img></a>
<a href="https://github.com/Imiloin/PacGoc/stargazers">
<img alt="stars" src="https://img.shields.io/github/stars/Imiloin/PacGoc?style=for-the-badge&color=orange&logo=apachespark&logoColor=red"></img></a>
<a href="https://www.python.org/">
<img alt="python" src="https://img.shields.io/badge/Python-%3E%3D3.10-blue?style=for-the-badge&color=blue&logo=Python&logoColor=blue"></img></a>
<a href="https://github.com/Imiloin/PacGoc?tab=GPL-3.0-1-ov-file#readme">
<img alt="license" src="https://img.shields.io/badge/LICENSE-GPL--3.0-yellow?style=for-the-badge&color=yellow&logo=googleslides&logoColor=yellow"></img></a>
</p>

<br>

<p>
如果本项目对您有帮助，不要忘记给它一个 ⭐️ 哦！
</p>

</div>

## Introduction

**PacGoc** 是 2024 年第八届全国大学生集成电路创新创业大赛紫光同创杯的全国(?)等奖作品。本项目包含了上位机部分（PC 端）使用的代码。

本项目构建了 Python 软件包 `pacgoc`，包含以下子模块：

+ `ans`：Acoustic Noise Suppression，声学降噪模块
+ `cls`：Classification，音频分类模块
+ `pcie_api`：PCIe API，PCIe 接收数据模块
+ `profiling`：Speaker Profiling，音频人物画像模块
    + `age_gender`：Age and Gender Prediction，预测年龄性别模块
    + `emotion`：Emotion Recognition，情感识别模块
+ `readwav`：Read WAV，流式读取 WAV 文件模块
+ `record`：Record，录音模块
+ `separation`：Audio Source Separation，音频源分离模块
+ `serial_api`：Serial API，串口通信模块
+ `spoof`：Spoof Detection，变声检测模块
+ `utils`：Utilities，工具函数模块
+ `sv`：Speaker Verification，声纹识别模块

项目使用 Python 语言编写。`pcie_api` 和 `serial_api` 模块须配合紫光同创盘古-50开发板（MES50HP）以及相应的硬件代码使用，其他模块可在普通 PC 上运行。

本项目还提供了比赛中使用的 Gradio Web 界面代码，保存在 `app` 目录下，用于统一展示项目的功能。

## What's New

+ 2024/06/25 本项目在初赛晋级 💪
+ 2024/07/28 本项目获得华东赛区分赛区决赛一等奖 🔥

## Installation

### Environment Setup

> [!NOTE]  
> 本项目的测试环境为 Ubuntu 20.04 + CUDA 11.8 + cuDNN 8.9.7 + Python 3.10 + PyTorch 2.3.0 + TensorFlow 2.16.1。

首先应确保安装了 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 、[cuDNN](https://developer.nvidia.com/cudnn) 及相应的 Nvidia 显卡驱动。本项目的测试版本为 CUDA 11.8 + cuDNN 8.9.7，其他版本的兼容性未知。

（推荐）安装 [Anaconda](https://www.anaconda.com/download)，便于环境配置。

您可能还需要安装一些依赖库：

```bash
sudo apt update
sudo apt install ffmpeg libsndfile1
```

如果您想快速复现比赛中使用的项目，在环境部署完毕后，可以直接查看[使用整合包的方法](#use-the-integration-package)，跳过下面的安装步骤。

### Install pacgoc package

创建一个新的 python 3.10 conda 环境并激活：

```bash
conda create -n pacgoc python=3.10
conda activate pacgoc
```

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合系统和 CUDA 版本的 PyTorch 安装命令。本项目的测试版本为 PyTorch 2.3.0，理论上高于 2.2.0 的版本均可。一个示例命令如下：

```bash
# change the CUDA version to the one you have
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or specify the version
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

安装其他依赖：

```bash
pip install -r requirements.txt
```

克隆本仓库：

```bash
git clone https://github.com/Imiloin/PacGoc.git
cd PacGoc
```

安装 `pacgoc` 包：

```bash
pip install -e .
```

### Installation for webui

在已有的环境下安装 Gradio：

```bash
# conda activate pacgoc
pip install -r requirements_app.txt
```

由于 [Gradio 的 issue #8160](https://github.com/gradio-app/gradio/issues/8160)，本项目使用的是 Gradio 4.16.0 版本。在此 issue 解决前，不建议使用较新的版本。

## Usage

### Use the pacgoc package

### Use the Gradio app

## Use the integration package

## FAQ

### 硬件部分的代码在哪？

本项目仅包含上位机部分的软件代码。硬件代码主要由 [@hosionn](https://github.com/hosionn) 负责，开源计划待定。

### 能在 Windows 上运行吗？

PCIe 模块使用了 Linux 版本的驱动，无法在 Windows 上使用。除此以外，`pacgoc` 包中的其他功能理论上可以在 Windows 上使用，但未经过详细的测试。

### Logo 有什么设计内涵？

Logo 整体为字母 “N” 的形状，与紫光同创标识相呼应。整体使用渐变色设计，优雅灵动，也具有创新活力。两侧伸出的圆形与内部线条构成音符的形状，代表了本项目音频处理的功能。

## Credits

> “If I have seen further, it is by standing on the shoulders of giants.”
> <div align="right">- Issac Newton</div>

本项目使用了 [pybind11](https://github.com/pybind/pybind11) 完成了 PCIe 与 Python 的交互，感谢 pybind11 的作者和贡献者。

本项目使用 [Gradio](https://gradio.app/) 轻松完成了 UI 界面，感谢 Gradio 的作者和贡献者。

本项目使用了很多软件包提供的 API，极大节省了开发时间。在此向这些开源项目的作者和贡献者表示感谢（排名不分先后）：

+ [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [modelscope](https://github.com/modelscope/modelscope)
+ [FunASR](https://github.com/modelscope/FunASR)
+ [Transformers](https://github.com/huggingface/transformers)

本项目还使用了很多开源项目提供的预训练模型，感谢这些项目的作者和贡献者（排名不分先后）：

+ [speech_frcrn_ans_cirm_16k](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k)
+ [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
+ [ced-base](https://huggingface.co/mispeech/ced-base)
+ [wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
+ [emotion2vec+large](https://huggingface.co/emotion2vec)
+ [zeroshot_asp_full](https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation)
+ [distilhubert](https://huggingface.co/ntu-spml/distilhubert)
+ [ecapatdnn_voxceleb12-16k](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#speaker-verification-models)
+ [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

## License

本项目编写的代码基于 [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) 协议开源。使用的预训练模型可能有不同的开源协议，具体请查看相应的项目。

## Disclaimer

The content provided in this repository is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
