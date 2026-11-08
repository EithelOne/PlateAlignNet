from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.demo_components import create_sample_image
from src.inference.factory import create_pipeline
from src.inference.schemas import InferenceOptions, InferenceStatus, RectificationMode


st.set_page_config(page_title="PlateAlignNet Demo", page_icon="🚘", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1320px;}
    [data-testid="stMetric"] {background: #111827; border: 1px solid #26354a; padding: 12px; border-radius: 12px;}
    .demo-badge {display:inline-block; padding:5px 10px; border-radius:999px; background:#4b3514; color:#ffd27a; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return create_pipeline("demo")


def image_to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("PlateAlignNet")
st.caption("检测 → STN 几何矫正 → LPRNet 免分割识别")
st.markdown('<span class="demo-badge">当前为启发式演示后端，不代表模型精度</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("推理设置")
    mode_label = st.selectbox("STN 模式", ["仿射（占位）", "透视（占位）", "关闭"])
    mode_map = {
        "仿射（占位）": RectificationMode.AFFINE,
        "透视（占位）": RectificationMode.PERSPECTIVE,
        "关闭": RectificationMode.OFF,
    }
    confidence = st.slider("检测置信度", 0.05, 0.90, 0.25, 0.05)
    st.divider()
    st.subheader("后续功能位置")
    st.checkbox("视频 / 摄像头", disabled=True)
    st.checkbox("批量识别", disabled=True)
    st.checkbox("无 STN / 仿射 / 透视并排消融", disabled=True)
    st.checkbox("历史记录与 CSV 导出", disabled=True)

demo_tab, architecture_tab, history_tab = st.tabs(["识别演示", "接口与扩展", "历史记录（预留）"])

with demo_tab:
    uploaded = st.file_uploader("上传车辆图片", type=["jpg", "jpeg", "png", "webp"])
    use_sample = st.toggle("使用内置样例", value=uploaded is None)
    source_image = None
    if uploaded is not None:
        try:
            source_image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"无法读取图片：{exc}")
    elif use_sample:
        source_image = create_sample_image()

    if source_image is None:
        st.info("上传一张图片，或开启内置样例。")
    else:
        options = InferenceOptions(
            rectification=mode_map[mode_label],
            confidence_threshold=confidence,
        )
        response = get_pipeline().run(source_image, options)
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            st.subheader("检测结果")
            st.image(response.annotated_image or source_image, use_container_width=True)
        with right:
            st.subheader("阶段结果")
            if response.status == InferenceStatus.ERROR:
                st.error(response.message)
            elif response.status == InferenceStatus.NO_PLATE:
                st.warning(response.message)
            else:
                plate = response.plates[0]
                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("车牌文本", plate.recognition.text)
                metric_b.metric("检测置信度", f"{plate.detection.confidence:.0%}")
                metric_c.metric("总耗时", f"{response.timing_ms['total']:.1f} ms")
                if plate.detection.is_fallback:
                    st.warning("未找到明显蓝/绿车牌区域，当前框为演示回退位置。")
                before, after = st.columns(2)
                before.image(plate.raw_plate, caption="原始车牌 ROI", use_container_width=True)
                after.image(plate.rectified_plate, caption=f"矫正输出 · {mode_label}", use_container_width=True)
                st.caption(
                    f"检测 {response.timing_ms['detect']:.2f} ms · "
                    f"矫正 {plate.timing_ms['rectify']:.2f} ms · "
                    f"识别 {plate.timing_ms['recognize']:.2f} ms"
                )

        if response.annotated_image is not None:
            download_image, download_json = st.columns(2)
            download_image.download_button(
                "下载可视化结果",
                data=image_to_png(response.annotated_image),
                file_name="platealignnet_result.png",
                mime="image/png",
                use_container_width=True,
            )
            download_json.download_button(
                "下载结构化结果",
                data=json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
                file_name="platealignnet_result.json",
                mime="application/json",
                use_container_width=True,
            )

with architecture_tab:
    st.subheader("替换真实模型时不改界面")
    st.code(
        """Detector.detect(image, confidence_threshold) -> list[Detection]
Rectifier.rectify(plate_image, mode, output_size) -> image
Recognizer.recognize(plate_image) -> Recognition

PlatePipeline.run(image, options) -> InferenceResponse""",
        language="python",
    )
    st.markdown(
        "真实模型接入顺序建议：先替换 YOLOv8 检测器，再接入 LPRNet，最后替换 STN，"
        "每一步都可复用当前结果对象、耗时统计、异常状态和下载功能。"
    )

with history_tab:
    st.info("此页保留给 SQLite/CSV 历史记录、筛选、错误案例画廊与实验统计。")
