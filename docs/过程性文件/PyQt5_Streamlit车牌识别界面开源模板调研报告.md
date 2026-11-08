# PyQt5 / Streamlit 车牌识别界面开源模板调研报告

> 调研日期：2026-11-08  
> 调研范围：GitHub 上可公开访问、与 PyQt5 或 Streamlit 车牌识别界面相关的开源项目  
> 本项目背景：PlateAlignNet（YOLOv8 检测 → STN 几何矫正 → LPRNet 免分割识别）

## 1. 调研目的与筛选方法

本次调研面向 PlateAlignNet 的可视化演示界面设计，重点考察以下内容：

1. 界面是否覆盖图片上传、原图展示、车牌区域展示和识别结果展示等核心交互；
2. 检测、裁剪、识别与界面之间如何组织，是否便于替换为本项目的 YOLOv8 + STN + LPRNet 推理管线；
3. 是否包含视频、摄像头、结果存储等可扩展功能；
4. 项目结构和许可证是否适合参考。

通过 GitHub 以 `license plate recognition PyQt5`、`license plate recognition Streamlit` 等关键词检索，并核对候选仓库的 README、目录结构、功能说明及许可证。最终选择 3 个侧重点互补的项目：一个经典 PyQt5 桌面实现、一个现代 YOLOv8 + OCR 的 PyQt5 实现，以及一个支持图片/视频流和数据库思路的 Streamlit 实现。

> 说明：以下结论来自仓库公开页面与 README，未对所有项目进行本地完整复现；Star 数、代码和依赖可能随时间变化。

## 2. 候选项目概览

| 项目                                                                                                                                                                                        | 界面技术            | 主要算法                            | 主要输入          | 参考价值                                        | 许可证信息                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------- | ----------------- | ----------------------------------------------- | ------------------------------------------------- |
| [DataXujing/vehicle-license-plate-recognition](https://github.com/DataXujing/vehicle-license-plate-recognition)                                                                             | PyQt5 / Qt Designer | OpenCV 传统定位 + 字符分割 + 双 SVM | 单张图片          | Qt 界面文件组织、经典完整流程、桌面打包         | MIT                                               |
| [zhongyanghan/License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8](https://github.com/zhongyanghan/License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8) | PyQt5               | YOLOv8 + PaddleOCR                  | 单张图片          | 与本项目最接近的现代桌面界面和检测/识别解耦方式 | AGPL-3.0                                          |
| [Parsa-SadeghiAsl/Iranian-LPR-Streamlit-App](https://github.com/Parsa-SadeghiAsl/Iranian-LPR-Streamlit-App)                                                                                 | Streamlit           | YOLOv8 + OCR                        | 图片、视频/实时流 | Web 演示、输入源切换、结果持久化思路            | 仓库标注 Apache-2.0，但 README 写 MIT，存在不一致 |

## 3. 参考项目详细分析

### 3.1 DataXujing/vehicle-license-plate-recognition

项目地址：<https://github.com/DataXujing/vehicle-license-plate-recognition>

#### 3.1.1 项目定位

该项目是较完整的中文车牌识别桌面软件示例。仓库同时包含 Qt Designer 的 `.ui` 文件、由界面文件生成的 Python 文件、主窗口逻辑、算法模块、资源文件以及打包配置，适合作为 PyQt5 工程结构和桌面发布流程的参考。

#### 3.1.2 界面设计

- 使用 PyQt5 构建桌面 GUI，并保留 `my_main_ui.ui` 与 `Ui_my_main_ui.py`，体现了“界面设计文件—生成代码—业务逻辑”分离的做法；
- 通过主窗口完成图像选择、识别触发、图像/结果显示；
- 使用 Qt 资源文件 `.qrc`、图标与图片资源统一管理界面素材；
- 仓库包含 PyInstaller 的 `.spec` 文件和 Windows 启动脚本，提供了从原型到可执行程序的工程化思路。

#### 3.1.3 算法设计

该项目采用传统计算机视觉与机器学习组合方案：

1. 对原图进行二值化、边缘检测，并利用车牌颜色等先验定位候选区域；
2. 裁剪车牌 ROI；
3. 使用直方图波峰/波谷完成字符分割；
4. 训练两个 SVM：一个识别省份简称，另一个识别字母和数字；
5. 将算法封装进 PyQt5 界面。

#### 3.1.4 可借鉴内容

- **借鉴界面工程结构，不照搬识别算法。** 本项目可保留 Qt Designer 的可视化布局方式，将算法入口替换为 PlateAlignNet 推理服务；
- 将 `Ui` 自动生成代码与事件逻辑分开，避免再次生成界面时覆盖业务代码；
- 使用 Qt Resource 管理 logo、占位图、按钮图标，便于统一视觉风格；
- 参考其 PyInstaller 打包流程，为课程答辩准备离线可执行演示程序；
- 传统算法可作为对比基线，用于说明 YOLOv8、STN 和 LPRNet 相比“颜色定位 + 字符分割 + SVM”的稳健性优势。

#### 3.1.5 局限

- 强依赖颜色、形状和字符分割，在复杂光照、倾斜、遮挡和低清晰度场景下泛化有限；
- 项目较早，依赖版本与现代 Python 环境可能不完全兼容；
- 界面架构偏同步调用，推理耗时时容易阻塞主线程，实际实现时应增加 `QThread` 或线程池。

### 3.2 zhongyanghan/License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8

项目地址：<https://github.com/zhongyanghan/License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8>

#### 3.2.1 项目定位

该项目使用 YOLOv8、PaddleOCR 和 PyQt5 实现中文车牌桌面识别，技术路线与 PlateAlignNet 最接近，适合重点参考其界面信息组织以及检测模块、识别模块和显示模块之间的衔接。

#### 3.2.2 界面设计

- 主区域采用直观的左右布局：左侧显示上传的车辆原图，右侧依次显示裁剪后的车牌、识别号码和车牌颜色；
- 使用“上传图片”作为主要操作入口，用户操作路径短，适合答辩演示；
- 未检测到车牌时显示占位图并给出“号码未识别出”的反馈，具备基本异常状态设计；
- 把原图、局部图和结构化文字结果同时呈现，便于用户核验识别是否正确。

#### 3.2.3 算法设计

1. 使用 YOLOv8 检测车牌，检测类别设置为蓝牌和绿牌；
2. 对检测框执行裁剪，获得车牌区域；
3. 将车牌图送入 PaddleOCR 提取字符；
4. 根据检测类别映射车牌颜色；
5. 在 PyQt5 中统一展示检测和识别结果。

仓库还提供训练脚本、YOLO 配置、数据集配置和预训练权重，算法流程清晰。但其界面目前主要支持单张图片，README 明确说明暂不支持视频流与批量推理。

#### 3.2.4 可借鉴内容

- **优先借鉴其信息层级：原图 → 车牌局部 → 号码 → 类型/置信度。** 这与本项目三阶段流水线天然匹配；
- 在右侧结果区新增“检测结果、STN 矫正前、STN 矫正后、最终车牌号”四层结果，使本项目的创新点可视化；
- 将算法接口抽象为统一结果对象，例如检测框、检测置信度、原始 ROI、矫正 ROI、字符序列、识别置信度和各阶段耗时，界面只负责渲染；
- 参考其“无检测结果”状态，同时补充模型未加载、输入格式错误、推理异常等状态；
- 将颜色/车牌类型作为辅助信息展示，但不要仅依赖检测类别判断全部牌照类型。

#### 3.2.5 局限与许可风险

- 当前仅支持单图，视频、摄像头、批量识别和历史记录尚未实现；
- OCR 在低分辨率和大倾角场景下会下降，恰好可由本项目的 STN 矫正模块形成差异化改进；
- 识别结果依赖扫描 `runs/detect/predict*` 目录，不利于并发和稳定调用。本项目应直接传递内存中的图像和结果，避免以临时目录作为模块接口；
- 项目采用 AGPL-3.0，Ultralytics YOLOv8 本身也涉及 AGPL/商业许可。可以学习设计思路，但若复制或修改代码并发布，需要严格核对开源义务。

### 3.3 Parsa-SadeghiAsl/Iranian-LPR-Streamlit-App

项目地址：<https://github.com/Parsa-SadeghiAsl/Iranian-LPR-Streamlit-App>

#### 3.3.1 项目定位

该项目是面向伊朗车牌的 Streamlit Web 应用，核心由 YOLOv8 检测和 OCR 识别组成。虽然车牌字符集和国内车牌不同，但它在浏览器演示、输入源扩展及结果数据管理方面具有较高参考价值。

#### 3.3.2 界面设计

- 基于 Streamlit 构建交互式 Web 页面，无需开发前后端即可快速形成浏览器演示；
- 支持上传图片，并提供实时视频流/离线视频的识别思路；
- 输入、模型推理和结果展示集中在单页应用中，部署后可通过局域网访问；
- 仓库将 `app.py`、`src/`、`settings/`、`models/`、`images/` 和 `videos/` 分开，结构比把所有逻辑写入单文件更易维护。

#### 3.3.3 算法设计

- YOLOv8 负责车牌区域检测；
- OCR 模型提取车牌字符；
- README 提到数据库集成，用于保存和管理识别记录；
- 支持图像和视频流场景，体现了“输入源适配器 + 通用推理管线 + 结果展示”的结构思路。

#### 3.3.4 可借鉴内容

- 若课程项目强调快速交付和跨设备展示，可优先制作 Streamlit 版本，减少桌面打包与系统兼容成本；
- 通过侧边栏集中放置模型权重、置信度阈值、IoU 阈值、是否启用 STN、仿射/透视模式等实验参数；
- 使用 Tab 分隔“识别演示、历史记录、性能统计、模型说明”，比单页堆叠更适合本项目；
- 使用 `st.cache_resource` 缓存 YOLOv8、STN 和 LPRNet 模型，避免 Streamlit 每次交互重新加载权重；
- 把图片、视频和摄像头统一为帧迭代接口，复用同一套检测—矫正—识别逻辑；
- 识别记录可存入 SQLite/CSV，字段包括时间、图片来源、车牌号、置信度、推理耗时和矫正模式，方便形成实验统计。

#### 3.3.5 局限与许可风险

- 目标是伊朗车牌，字符集、版式和 OCR 模型不能直接用于中文车牌；
- Streamlit 在高帧率视频、复杂桌面交互和精细 UI 控制方面不如 PyQt5；
- 仓库页面将许可证标为 Apache-2.0，而 README 文本写 MIT，二者不一致。若要复用代码，应以仓库实际 `LICENSE` 文件及具体提交版本为准，不能仅依据 README。

## 4. 对本项目的综合借鉴方案

### 4.1 推荐界面布局

建议采用“左侧输入与原图、右侧阶段结果、底部状态与耗时”的结构：

```text
┌──────────────────────────┬──────────────────────────┐
│ 输入源与原图              │ 识别结果                  │
│                          │ 1. YOLO 检测框/车牌 ROI   │
│ [选择图片/视频/摄像头]    │ 2. STN 矫正前后对比       │
│ [开始] [停止] [保存结果]  │ 3. 车牌号码与置信度       │
├──────────────────────────┴──────────────────────────┤
│ 模型状态｜检测/矫正/识别耗时｜总 FPS｜错误提示       │
└─────────────────────────────────────────────────────┘
```

界面应突出本项目的 STN 创新：矫正前、仿射矫正后或透视矫正后的车牌图应并排显示，并允许关闭 STN，以便现场展示消融效果。

### 4.2 推荐功能优先级

#### 第一阶段：答辩所需最小可用版本

- 单张图片上传与识别；
- 原图检测框、原始车牌 ROI、STN 矫正结果和最终号码展示；
- 检测置信度、识别置信度、分阶段耗时展示；
- 无车牌、模型缺失和推理失败的清晰提示；
- 一键保存可视化结果。

#### 第二阶段：增强演示能力

- 批量图片和文件夹推理；
- 视频文件/摄像头识别；
- 识别历史、筛选和 CSV 导出；
- STN 开关及仿射/透视模式切换；
- CCPD 挑战子集或典型样例快捷选择。

#### 第三阶段：实验展示

- 实时显示各阶段耗时、FPS 和显存/设备信息；
- 同一输入下“无 STN / 仿射 STN / 透视 STN”的并排对比；
- 统计检测成功率、整牌准确率和字符准确率；
- 提供错误案例画廊，展示模糊、倾斜、旋转和强光场景。

### 4.3 推荐软件结构

无论选择 PyQt5 还是 Streamlit，都不应把模型加载和推理细节直接写入按钮回调。建议采用以下分层：

```text
ui/
├── app.py                 # PyQt5 或 Streamlit 入口
├── views/                 # 页面/组件
└── presenters/            # 将推理结果转换为界面数据

src/inference/
├── pipeline.py            # 检测 → 矫正 → 识别统一编排
├── schemas.py             # 输入、阶段结果与错误结构
└── sources.py             # 图片、视频、摄像头输入适配
```

统一推理结果建议至少包含：

```python
{
    "bbox": [x1, y1, x2, y2],
    "detection_confidence": 0.99,
    "raw_plate": "...",
    "rectified_plate": "...",
    "plate_text": "京A12345",
    "recognition_confidence": 0.97,
    "timing_ms": {"detect": 12.3, "rectify": 2.1, "recognize": 6.8}
}
```

PyQt5 版本应将模型推理放入 `QThread`/线程池，防止窗口卡死；Streamlit 版本应缓存模型，并用 session state 管理当前输入、运行状态和历史结果。

## 5. PyQt5 与 Streamlit 选型建议

| 维度           | PyQt5                            | Streamlit                       |
| -------------- | -------------------------------- | ------------------------------- |
| 开发速度       | 中等，需要处理布局、线程和信号槽 | 快，适合快速搭建演示            |
| 界面控制力     | 强，可制作较完整的桌面软件       | 中等，受 Web 组件与刷新机制限制 |
| 视频实时性     | 较好，适合本地摄像头与逐帧刷新   | 一般，需要额外处理视频流        |
| 部署方式       | 运行 Python 或打包桌面程序       | 启动服务后用浏览器访问          |
| 跨设备演示     | 较弱，需安装或分发程序           | 较强，局域网访问方便            |
| 与实验面板结合 | 可实现，但开发量较大             | 很适合参数控件、图表和数据表    |
| 推荐用途       | 最终桌面演示、实时摄像头         | 快速原型、实验对比、远程展示    |

综合课程项目目标，建议先实现 **Streamlit 最小可用版本**，快速贯通 YOLOv8 → STN → LPRNet 并展示消融参数；若时间允许，再用 PyQt5 实现适合答辩现场的桌面版本。若最终只维护一套界面：偏重算法实验展示时选 Streamlit，偏重“完整软件系统”和实时摄像头时选 PyQt5。

## 6. 结论

三个项目分别提供了不同层面的参考：

- DataXujing 项目适合参考 PyQt5 工程组织、Qt Designer 和桌面打包；
- zhongyanghan 项目与本项目技术路线最接近，适合参考原图—车牌局部—号码/类型的结果布局；
- Iranian-LPR-Streamlit-App 适合参考 Web 化演示、多输入源和识别记录管理。

本项目不宜直接照搬任何一个模板，而应组合三者优点：以现代检测/识别界面的信息层级为基础，引入 Streamlit 的快速实验面板或 PyQt5 的桌面交互，同时把 **STN 矫正前后对比、消融开关和分阶段性能指标**作为差异化核心。代码复用前必须再次核对目标仓库具体版本的许可证，尤其注意 AGPL-3.0 传播义务以及第三个项目许可证说明不一致的问题。

## 7. 参考资料

1. DataXujing. *vehicle-license-plate-recognition*. GitHub. <https://github.com/DataXujing/vehicle-license-plate-recognition>（访问日期：2026-11-08）。
2. zhongyanghan. *License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8*. GitHub. <https://github.com/zhongyanghan/License-Plate-Detection-and-Number-Recognition-System-Based-on-Yolov8>（访问日期：2026-11-08）。
3. Parsa-SadeghiAsl. *Iranian-LPR-Streamlit-App*. GitHub. <https://github.com/Parsa-SadeghiAsl/Iranian-LPR-Streamlit-App>（访问日期：2026-11-08）。
4. Ultralytics. *YOLO Documentation*. <https://docs.ultralytics.com/>（访问日期：2026-11-08）。
5. PaddlePaddle. *PaddleOCR*. GitHub. <https://github.com/PaddlePaddle/PaddleOCR>（访问日期：2026-11-08）。

