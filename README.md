# PlateAlignNet

> **牌正识**：基于 STN 矫正与端到端免分割识别的中文车牌识别系统
> YOLOv8 Detection · STN Rectification · LPRNet Recognition · CCPD Benchmark

---

## 1. 项目简介

PlateAlignNet 是一个面向中文车牌的端到端识别系统，采用 **"检测 → 矫正 → 识别"** 三阶段级联架构：

- **检测**：YOLOv8n 在 CCPD2019 上完成车牌定位；
- **矫正**：空间变换网络（STN）对倾斜、旋转车牌进行几何矫正；
- **识别**：LPRNet 基于 CTC 损失实现免字符分割的端到端序列识别。

本项目的核心研究切入点是 **STN 矫正模块的系统性消融研究**：对比仿射变换与透视变换、定位网络深度、STN 插入位置、端到端联合训练与分阶段训练等设计维度对最终识别精度的影响，并在 CCPD 的倾斜、旋转、模糊、亮度等挑战子集上进行细粒度性能分析。

**目标指标**：检测 mAP@0.5 ≥ 98%，端到端识别准确率 ≥ 97%。

---

## 2. 核心特性

- 中文蓝色车牌 + 新能源绿色车牌支持
- STN 矫正模块可插拔，支持仿射 / 透视两种变换
- 端到端联合训练与分阶段训练双模式
- 在 CCPD-DB / Blur / FN / Rotate / Tilt / Challenge 六个挑战子集上细粒度评估
- PyQt5 / Streamlit 可视化演示界面
- 完整消融实验配置与复现脚本

---

## 3. 技术架构

```text
输入图像
   │
   ▼
┌─────────────┐
│  YOLOv8n    │  车牌检测（边界框）
└─────────────┘
   │
   ▼
┌─────────────┐
│     STN     │  几何矫正（仿射 / 透视）
└─────────────┘
   │
   ▼
┌─────────────┐
│   LPRNet    │  CTC 序列识别
└─────────────┘
   │
   ▼
车牌字符序列
```

---

## 4. 仓库结构

```text
PlateAlignNet/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/                 # 超参数配置
│   ├── yolov8_plate.yaml
│   ├── lprnet.yaml
│   └── stn.yaml
├── src/
│   ├── data/                # CCPD 解析与数据增强
│   ├── models/              # detector / stn / lprnet
│   ├── train/               # 训练脚本
│   ├── eval/                # 评估脚本
│   ├── inference/           # 推理管线
│   └── utils/               # 日志、指标、可视化
├── scripts/                 # 一键脚本
├── docs/                    # 开题报告、技术方案、实验记录
├── weights/                 # 本地权重（gitignore）
├── outputs/                 # 训练日志与评估结果
├── ui/                      # 可视化界面
└── tests/                   # 单元测试
```

---

## 5. 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0（CUDA 11.8 推荐）
- Ultralytics ≥ 8.0
- OpenCV、NumPy、Pillow、PyYAML、TensorBoard
- PyQt5 或 Streamlit（界面演示）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 6. 快速开始

### 6.0 先运行可扩展 Demo

当前仓库提供一个不依赖模型权重的 Streamlit 演示后端，用于验证界面与“检测 → 矫正 → 识别”流水线。演示结果会明确标注为占位结果，不代表真实模型精度。

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

详细说明见 `docs/DEMO.md`。真实 YOLOv8、STN 和 LPRNet 后续可分别实现统一接口并注册到推理工厂，无需重写页面。

### 6.1 克隆仓库

```bash
git clone https://github.com/your-team/PlateAlignNet.git
```

```bash
cd PlateAlignNet
```

### 6.2 数据准备

下载 CCPD2019 数据集并放置到 `data/CCPD2019/`，然后运行解析脚本转换为 YOLO 格式：

```bash
python src/data/ccpd_parser.py --root data/CCPD2019 --out data/ccpd_yolo
```

### 6.3 训练检测模型

```bash
python src/train/train_detector.py --config configs/yolov8_plate.yaml
```

### 6.4 训练识别模型

```bash
python src/train/train_recognizer.py --config configs/lprnet.yaml
```

### 6.5 端到端联合训练

```bash
python src/train/train_end2end.py --config configs/stn.yaml
```

### 6.6 评估

```bash
python src/eval/evaluate.py --split test --subset base
```

```bash
python src/eval/evaluate.py --split test --subset tilt
```

### 6.7 推理演示

```bash
python src/inference/pipeline.py --source demo.jpg
```

### 6.8 启动可视化界面

```bash
python scripts/run_demo.py
```

---

## 7. Git 协作规范

### 7.1 首次配置（每人一次）

```bash
git config --global user.name "你的名字"
```

```bash
git config --global user.email "你的邮箱"
```

### 7.2 首次克隆仓库

```bash
git clone https://github.com/your-team/PlateAlignNet.git
```

### 7.3 日常拉取最新代码（每次开工前）

```bash
git pull origin main
```

### 7.4 查看当前修改状态

```bash
git status
```

### 7.5 添加修改到暂存区

```bash
git add .
```

### 7.6 提交修改

```bash
git commit -m "feat: 完成 STN 仿射变换模块"
```

提交信息建议遵循 `type: 描述` 格式，type 可选：`feat` / `fix` / `docs` / `refactor` / `exp`（实验）。

### 7.7 推送到远程仓库

```bash
git push origin main
```

### 7.8 创建功能分支（推荐，避免直接改 main）

```bash
git checkout -b feature/stn-rectification
```

### 7.9 切换回主分支

```bash
git checkout main
```

### 7.10 合并功能分支

```bash
git merge feature/stn-rectification
```

### 7.11 推送功能分支到远程

```bash
git push origin feature/stn-rectification
```

### 7.12 冲突处理流程

查看冲突文件：

```bash
git status
```

手动编辑冲突文件后，重新添加：

```bash
git add 冲突文件路径
```

完成合并提交：

```bash
git commit -m "merge: 解决 STN 分支冲突"
```

### 7.13 撤销未提交的修改

```bash
git checkout -- 文件路径
```

### 7.14 查看提交历史

```bash
git log --oneline --graph
```

### 7.15 协作约定

- `main` 分支保持可运行，不直接提交实验性代码；
- 每位成员在自己的 `feature/xxx` 分支开发，完成后发起 Pull Request；
- 提交前必须运行 `python -m pytest tests/` 确保基础测试通过；
- 权重文件、数据集、日志不进入 Git，已在 `.gitignore` 中忽略。

---

## 8. 文档索引

|     文档     |          路径          |             说明             |
| :----------: | :--------------------: | :--------------------------: |
|   开题报告   |   `docs/开题报告.md`   |   研究背景、目标、技术路线   |
|   技术方案   |   `docs/技术方案.md`   | 模型结构、训练策略、消融设计 |
|   实验记录   |   `docs/实验记录.md`   |      各轮实验配置与结果      |
| 挑战子集分析 | `docs/挑战子集分析.md` |   CCPD 六个子集细粒度结果    |

---

## 9. 实验进度

|              阶段              |   时间   |  状态  |
| :----------------------------: | :------: | :----: |
|            开题报告            | 第 7 周  | 已完成 |
| 进度检查（技术方案、实验方案） | 第 12 周 | 进行中 |
|            项目汇报            | 第 16 周 | 待开始 |

---

## 10. 参考文献

1. Xu Z, Yang W, Meng A, et al. Towards end-to-end license plate detection and recognition: A large dataset and baseline. ECCV, 2018.
2. Zherzdev S, Gruzdev A. LPRNet: License plate recognition via deep neural networks. arXiv:1806.10447, 2018.
3. Jaderberg M, Simonyan K, Zisserman A, et al. Spatial transformer networks. NeurIPS, 2015.
4. Xiong W, Cao L, Yan D, et al. License plate recognition methodology in complex scenarios based on CSCM-YOLOv8 and CSM-LPRNet. PLoS One, 2026.
5. Deep learning algorithms for license plate recognition: A review. Neural Networks, 2026.

---

## 11. License

本项目采用 MIT License，详见 `LICENSE` 文件。
