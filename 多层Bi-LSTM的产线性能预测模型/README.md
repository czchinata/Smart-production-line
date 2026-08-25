# **制造业生产指标预测系统 — 使用说明**

## **一、项目概述**

本项目是一个完整的制造业时序数据预测系统，采用 **CNN + BiLSTM + Attention** 混合深度学习架构，用于预测生产时间、设备利用率、质量波动、单位成本和交付能力等 5 项关键指标。

## **二、文件说明**

表格

| 文件名             | 功能定位                                                     | 是否直接运行 |
| :----------------- | :----------------------------------------------------------- | :----------- |
| `generate_data.py` | 合成数据生成器，模拟制造业物理规律生成时序数据               | ✅ 是         |
| `model.py`         | 神经网络模型定义（CNN + BiLSTM + Attention），被其他文件导入 | ❌ 否         |
| `optimize.py`      | 自适应粒子群优化（PSO）超参数搜索器                          | ✅ 是（可选） |
| `train.py`         | 主程序：训练、评估、可视化、结果导出                         | ✅ 是         |

## **三、环境配置**

### **3.1 创建虚拟环境**

bash

```
# 在项目根目录创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# macOS / Linux 激活
source .venv/bin/activate
```

### **3.2 安装依赖**

bash

```
pip install torch numpy pandas scikit-learn matplotlib tqdm
```

> **GPU 用户**：请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本安装。

### **3.3 PyCharm 配置**

1. 打开 PyCharm → **File → Settings → Project → Python Interpreter**
2. 点击齿轮图标 → **Add** → 选择 **Existing environment**
3. 路径指向项目目录下的 `.venv/Scripts/python.exe`（Windows）或 `.venv/bin/python`（Mac/Linux）

## **四、运行流程（严格按顺序）**

### **步骤 1：生成数据**

bash

```
python generate_data.py
```

- **输出**：`data/manufacturing_data.csv`（12,000 条记录）
- **耗时**：约 5~10 秒
- **说明**：生成包含 12 个特征和 5 个目标的模拟制造业数据

### **步骤 2：超参数优化（可选）**

bash

```
python optimize.py
```

- **输出**：控制台打印最优超参数组合及验证损失
- **耗时**：10~30 分钟（视搜索空间而定）
- **说明**：使用自适应 PSO 算法搜索最佳学习率、隐藏层维度、注意力头数等

### **步骤 3：训练与评估（主程序）**

bash

```
python train.py
```

- **输出**：
  - `checkpoints/best_model.pth` — 最佳模型权重
  - `results/metrics.csv` — 评估指标
  - `results/predictions.csv` — 预测结果
  - `results/prediction_curves.png` — 预测对比图
  - `results/learning_curve.png` — 学习曲线图
- **说明**：自动加载数据 → 标准化 → 训练 → 早停 → 测试集评估 → 导出结果

## **五、项目目录结构**

text

```
manufacturing-prediction/
├── generate_data.py        # 数据生成
├── model.py                # 模型定义（被导入，不直接运行）
├── optimize.py             # 超参数优化
├── train.py                # 主训练程序
├── data/
│   └── manufacturing_data.csv
├── checkpoints/
│   └── best_model.pth
└── results/
    ├── metrics.csv
    ├── predictions.csv
    ├── prediction_curves.png
    └── learning_curve.png
```

## **六、PyCharm 调试技巧**

表格

| 操作         | 方法                               |
| :----------- | :--------------------------------- |
| 设置断点     | 点击代码行号左侧                   |
| 调试运行     | 右键 → **Debug 'train'**           |
| 查看变量     | 底部 **Variables** 面板            |
| 修改运行参数 | 右上角运行配置 → **Parameters** 栏 |
| 查看图片     | 项目面板中双击 `.png` 文件         |

## **七、常见问题**

表格

| 问题                                           | 原因                        | 解决方案                           |
| :--------------------------------------------- | :-------------------------- | :--------------------------------- |
| `ModuleNotFoundError: No module named 'torch'` | 解释器未指向虚拟环境        | 检查 Python Interpreter 设置       |
| `FileNotFoundError: data/...csv`               | 未先运行 `generate_data.py` | 先执行步骤 1                       |
| `CUDA out of memory`                           | GPU 显存不足                | 减小 `batch_size` 或改用 CPU       |
| 训练 Loss 不下降                               | 学习率过大 / 数据异常       | 降低学习率，检查数据标准化         |
| `model.py` 运行报错                            | 该文件无 `__main__` 入口    | 不要直接运行，它由 `train.py` 导入 |

## **八、快速验证**

首次使用时，建议临时将 `train.py` 中的 `epochs` 改为 `5`，快速验证全流程无报错后再恢复正式参数。