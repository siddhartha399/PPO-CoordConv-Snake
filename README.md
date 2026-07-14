# 🐍 PPO-CoordConv-Snake

### Fully GPU-Accelerated PPO Training for Snake with 4,096 Parallel Environments

Train a high-performing Snake agent entirely on the GPU using PPO, GAE, CoordConv, and a spatially-preserving convolutional architecture.

<p>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch" alt="PyTorch">
  </a>
  <a href="./notebooks/PPO_Snake_Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>

**Train a Snake agent entirely on the GPU with 4,096 parallel environments, PPO, GAE, CoordConv, and a zero-downsampling architecture designed for grid-based reinforcement learning.**

</div>

---

## 🎮 Demo

<p align="center">
  <img src="assets/snake_demo.gif" alt="PPO-CoordConv-Snake Demo" width="900">
</p>

<p align="center">
  <i>Agent performance after training on a single NVIDIA T4 GPU.</i>
</p>

---

## ✨ Why PPO-CoordConv-Snake?

Most Snake reinforcement learning projects are bottlenecked by CPU simulation, low environment throughput, or architectures that destroy spatial information through aggressive downsampling.

PPO-CoordConv-Snake was designed around a simple idea:

> Keep everything on the GPU and preserve spatial information until the final decision layer.

### Key Features

* ⚡ **4,096 parallel environments running directly in VRAM**
* 🧠 **Proximal Policy Optimization (PPO)**
* 📈 **Generalized Advantage Estimation (GAE)**
* 🗺️ **CoordConv spatial encoding**
* 🔍 **Zero-downsampling convolutional backbone**
* 🚀 **PyTorch 2.x compilation acceleration**
* 🎮 Trains a strong Snake policy on a single T4 GPU
* 📓 Ready-to-run Google Colab notebook

---

## 📊 Results

| Metric                | Value          |
| --------------------- | -------------- |
| Mean Score            | **86.151**     |
| Max Score             | **87**         |
| Grid Size             | **9 × 10**     |
| Parallel Environments | **4,096**      |
| Hardware              | **NVIDIA T4**  |
| Training Time         | **< 10 Hours** |
| Algorithm             | **PPO + GAE**  |

These results were achieved using a single T4 GPU while keeping environment simulation and policy optimization on the GPU.

---

## 🏗 Architecture

The architecture is designed to preserve spatial structure throughout the network while maintaining high throughput.

### 🗺️ CoordConv

Instead of forcing the network to infer absolute position, static X and Y coordinate channels are injected directly into the observation tensor.

This allows the policy to reason about location explicitly while improving sample efficiency.

### 🔍 Zero-Downsampling Backbone

Many convolutional reinforcement learning agents reduce spatial resolution using pooling or strided convolutions.

PPO-CoordConv-Snake preserves the full grid resolution throughout feature extraction using dilated residual blocks.

### ⚡ GPU-Native Environment Simulation

Environment stepping occurs directly in VRAM, eliminating host-device transfer bottlenecks and enabling thousands of parallel simulations.

### 🚀 PyTorch 2.x Acceleration

Critical training components leverage PyTorch compilation to reduce overhead and improve runtime performance.

---

## 🌊 Model Flow

```text
Input Tensor [B=4096, C=7, H=9, W=10]
│
├── Head
├── Body
├── Tail
├── Food
├── Danger
├── Coord Y
└── Coord X
│
▼
Conv2D + ReLU
│
▼
Dilated Residual Blocks
│
▼
─────────────────────────────
│                           │
▼                           ▼
Actor Head              Critic Head
│                           │
▼                           ▼
Action Logits          State Value
```

---

## 🚀 Quick Start

### Clone the Repository

```bash
git clone https://github.com/siddhartha399/PPO-CoordConv-Snake.git
cd PPO-CoordConv-Snake
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train an Agent

```bash
python train.py
```

### Evaluate a Trained Agent

```bash
python evaluate.py --checkpoint models/best_model_v1.pt --output snake_evaluation.mp4
```

---

## 📓 Google Colab

Run the project directly from Colab with no local CUDA setup required.

```text
notebooks/PPO_Snake_Colab.ipynb
```

---

## 📂 Repository Structure Nurture

```text
PPO-CoordConv-Snake/
├── assets/
│   └── snake_demo.gif
│
├── models/
│   └── best_model_v1.pt
│
├── notebooks/
│   ├── PPO_Snake_Colab.ipynb
│   └── Dashboard_Visualizer.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── env.py
│   └── utils.py
│
├── .gitignore
├── evaluate.py
├── train.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔬 Technical Stack

* PyTorch 2.x
* PPO
* GAE
* CoordConv
* Dilated Residual Networks
* NumPy
* Pillow
* TensorBoard
* imageio

---

## 🎯 Project Goals

PPO-CoordConv-Snake explores how GPU-native environment simulation and spatially-aware neural architectures can improve efficiency in grid-based reinforcement learning tasks.

The project is intended for:

* Reinforcement Learning enthusiasts
* Machine Learning engineers
* Students learning PPO
* Researchers exploring spatial representations
* Developers interested in GPU-first RL systems

---

## 🤝 Contributing

Contributions, issues, feature requests, and improvements are welcome.

If you discover a bug or have an idea for improving performance, feel free to open an issue or submit a pull request.

---

## 📜 License

Released under the MIT License.

See the LICENSE file for details.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star.

It helps others discover the project and supports future development.

</div>
