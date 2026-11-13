# PlateAlignNet

> **牌正识**：面向中文车牌的检测—矫正—识别课程项目
> 目标架构：YOLOv8 Detection · STN Rectification · LPRNet Recognition · CCPD Benchmark

---

## 1. 项目简介

PlateAlignNet 是一个面向中文车牌识别的课程项目，目标是构建 **"检测 → 矫正 → 识别"** 三阶段级联系统：

- **检测**：YOLOv8n 在 CCPD2019 上完成车牌定位；
- **矫正**：空间变换网络（STN）对倾斜、旋转车牌进行几何矫正；
- **识别**：LPRNet 基于 CTC 损失实现免字符分割的端到端序列识别。

本项目的核心研究切入点是 **STN 矫正模块的系统性消融研究**：对比仿射变换与透视变换、定位网络深度、STN 插入位置、端到端联合训练与分阶段训练等设计维度对最终识别精度的影响，并在 CCPD 的倾斜、旋转、模糊、亮度等挑战子集上进行细粒度性能分析。

**目标指标**：检测 mAP@0.5 ≥ 98%，端到端识别准确率 ≥ 97%。

---

## 2. 当前实现

- Streamlit 单图演示界面，包含图片上传、内置样例、阶段结果和 JSON / 图片下载。
- 推理流水线定义了检测、矫正、识别协议；当前工厂只注册 `demo-heuristic` 演示后端。检测使用颜色启发式，识别文本为占位结果，矫正只做尺寸归一化，不代表真实模型效果。
- 已有 LPRNet、CTC 编解码及网络规格实现，但真实数据解析/数据集模块、完整训练评估流程、YOLOv8 检测器和 STN 模型尚未在当前仓库中提供或接入。
- CCPD 蓝牌、新能源车牌和 STN 消融研究属于项目目标；当前演示不应据此宣称已支持真实识别或已完成相关实验。

---

## 3. 目标技术架构

以下是项目设计目标，并非当前仓库已完整实现的模型链路：

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
├── 可写进报告或PPT中的思考和问题.md
├── configs/
│   └── demo.toml            # 演示界面与推理参数
├── src/
│   ├── inference/            # 推理协议、演示组件、流水线与结果可视化
│   ├── models/               # LPRNet 与网络规格
│   ├── train/                # 识别器小样本过拟合检查入口
│   └── utils/                # 车牌字符表与 CTC 编解码
├── scripts/
│   └── run_demo.py           # 启动 Streamlit 演示
├── ui/
│   └── app.py                # Streamlit 页面
├── tests/                    # 单元测试
└── docs/
    ├── DEMO.md               # 演示说明与分阶段验收清单
    ├── 过程性文件/            # 界面模板调研报告
    └── 提交报告/              # 开题报告、进展报告及模板
```

数据集、模型权重和运行输出应放在本地 `data/`、`weights/`、`outputs/` 目录；这些目录已由 `.gitignore` 排除，不随仓库提交。注意：当前版本没有 `src/data/` 实现，不能据目录名或测试文件推断 CCPD 解析及训练链路已可运行。

---

## 5. 环境依赖

演示后端依赖 Python、NumPy、Pillow 和 Streamlit，版本范围见 `requirements.txt`。当前训练入口及 LPRNet 实现还依赖 PyTorch，但 PyTorch 未列入演示依赖文件；YOLOv8、STN 所需依赖也尚未纳入当前可运行流程。

安装演示依赖：

```bash
pip install -r requirements.txt
```

---

## 6. 快速开始

### 6.1 克隆仓库

```bash
git clone https://github.com/your-team/PlateAlignNet.git
cd PlateAlignNet
```

### 6.2 启动演示

当前可运行入口是不依赖模型权重的 Streamlit 演示后端，用于体验界面和推理接口。演示结果会明确标注为占位结果，不代表真实模型精度。

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

详细说明见 [`docs/DEMO.md`](docs/DEMO.md)。真实检测器、矫正器和识别器可按推理协议实现后注册到工厂。

### 6.3 检查 LPRNet 结构（可选）

安装与本机环境匹配的 PyTorch 后，可运行网络规格检查和模型规格一致性检查：

```bash
python -m src.models.lprnet_spec
python -m src.models.lprnet
```

这两项检查不等同于真实数据训练或识别评估。

当前仓库尚无可用的 CCPD 解析、检测训练、STN 训练、端到端训练或评估命令。`src/train/train_recognizer.py` 目前只提供小样本过拟合检查入口，并引用尚未提交的 `src.data` 模块；在该数据模块补齐前，此入口不可运行。不要将演示占位结果或模型结构自检当作实验结果。

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
- 提交前应运行 `python -m pytest tests/`；当前仓库仍有测试引用缺失的 `src.data` 模块，测试尚不能作为全量通过的验收依据；
- 权重文件、数据集、日志不进入 Git，已在 `.gitignore` 中忽略。

---

## 8. 文档索引

|      文档      |                                                                路径                                                                | 说明                                                               |
| :------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------- |
|    演示说明    |                                                   [`docs/DEMO.md`](docs/DEMO.md)                                                   | 当前演示实现、启动方法、接口扩展与验收清单                         |
|    开题报告    |                                         `docs/提交报告/人工智能导论_开题报告.doc` / `.pdf`                                         | 课程项目开题材料                                                   |
|    进展报告    |                    [`docs/提交报告/人工智能导论_进展报告.md`](docs/提交报告/人工智能导论_进展报告.md) / `.docx`                    | 项目任务、技术方案和实验方案；进展与待解决问题部分留待实际结果补充 |
|    报告模板    |                                           `docs/提交报告/人工智能导论_进展报告模板.doc`                                            | 进展报告模板                                                       |
|    调研报告    | [`docs/过程性文件/PyQt5_Streamlit车牌识别界面开源模板调研报告.md`](docs/过程性文件/PyQt5_Streamlit车牌识别界面开源模板调研报告.md) | PyQt5 / Streamlit 界面模板调研                                     |
| 思考与问题记录 |                               [`可写进报告或PPT中的思考和问题.md`](可写进报告或PPT中的思考和问题.md)                               | 开发过程中的问题、技术思考及修改记录                               |

---

## 9. 当前状态

| 模块                                  | 当前状态                               |
| :------------------------------------ | :------------------------------------- |
| Streamlit 单图演示与统一推理接口      | 已实现，使用启发式/占位演示组件        |
| 字符表、CTC 编解码、LPRNet 结构与模型 | 已实现基础代码；模型自检需安装 PyTorch |
| CCPD 数据解析与识别数据集             | 当前仓库缺少 `src/data/` 实现          |
| YOLOv8 检测器、真实 STN、模型推理适配 | 尚未接入                               |
| 正式训练、统一评估与消融实验          | 尚未形成可运行完整流程或实验结果       |

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
