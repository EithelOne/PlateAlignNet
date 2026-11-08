# PlateAlignNet Demo 说明

## 当前实现

Demo 贯通了单张图片的完整交互链路：图片输入、候选车牌检测、ROI 裁剪、STN 模式选择、归一化输出、车牌文本、置信度、分阶段耗时以及结果下载。

当前仓库没有数据集和模型权重，因此默认使用 `demo-heuristic` 后端：检测由蓝色或绿色区域启发式完成，无法找到候选区域时使用中央回退框；识别文本是确定性的演示占位结果；“仿射”和“透视”模式目前只执行 ROI 尺寸归一化。页面会持续显示演示后端提示，避免把占位结果误认为真实模型精度。

## 启动

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
```

也可以直接运行：

```bash
python -m streamlit run ui/app.py
```

## 扩展位置

真实模型只需实现三个协议：

- `Detector.detect`：接收完整图像，返回边界框、置信度和车牌类型；
- `Rectifier.rectify`：接收车牌 ROI、变换模式和输出尺寸，返回矫正图；
- `Recognizer.recognize`：接收车牌图，返回文本和置信度。

实现后在 `src/inference/factory.py` 注册新的后端。界面、异常状态、阶段耗时、结构化结果和下载功能均可继续复用。

后续功能已在界面侧边栏和标签页预留：视频/摄像头、批量推理、三种 STN 模式并排消融、历史记录、CSV 导出、错误案例画廊和实验统计。
