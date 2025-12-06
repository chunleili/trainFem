# 使用神经网络替代FEM能量最小化

## 项目概述

这个项目展示了如何使用PyTorch训练神经网络来替代传统的有限元法(FEM)能量最小化过程。

### 核心思想

1. **数据生成**：使用传统FEM求解器（共轭梯度法）生成大量训练数据
2. **网络训练**：训练神经网络学习从初始状态到优化状态的映射
3. **快速推理**：使用训练好的网络进行快速模拟

## 文件说明

- `tiFEM` - 传统FEM求解器（基于Taichi）
- `train_fem.py` - 神经网络训练脚本
- `inference_fem.py` - 使用神经网络进行推理的可视化

## 使用方法

### 1. 安装依赖

```bash
pip install torch taichi numpy
```

### 2. 训练模型

```bash
python train_fem.py
```

这将：
- 生成100*100个训练样本（使用FEM求解器），保存到 `fem_dataset.npz`
- 训练并保存最佳模型到 `best_fem_model.pth`

训练过程约需要几分钟，取决于硬件配置。

### 3. 可视化推理

```bash
python inference_fem.py
```

这将打开一个3D窗口，显示使用神经网络求解的FEM模拟。
