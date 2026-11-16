from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.demo_components import create_sample_image
from src.inference.factory import create_pipeline
from src.inference.real_backend import (
    STNRectifier,
    ManifestDetector,
    ManualBoxDetector,
    RealPlatePipeline,
    TorchRecognizer,
    load_recognizer,
)
from src.inference.schemas import InferenceOptions, InferenceStatus, RectificationMode

DEVICE = torch.device("cpu")
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "ccpd_manifest_base" / "samples.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs"
PLATE_INPUT_SIZE = (188, 48)


st.set_page_config(page_title="PlateAlignNet Demo", page_icon="🚘", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1320px;}
    [data-testid="stMetric"] {background: #111827; border: 1px solid #26354a; padding: 9px 12px; border-radius: 12px; margin-bottom: 8px;}
    [data-testid="stMetricLabel"] {font-size: 0.78rem;}
    [data-testid="stMetricValue"] {font-size: 1.25rem; line-height: 1.35; overflow-wrap: anywhere;}
    .demo-badge {display:inline-block; padding:5px 10px; border-radius:999px; background:#4b3514; color:#ffd27a; font-size:13px;}
    .real-badge {display:inline-block; padding:5px 10px; border-radius:999px; background:#123c2a; color:#6ff0a8; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_demo_pipeline():
    return create_pipeline("demo")


@st.cache_resource
def get_real_model(checkpoint_path: str):
    """按权重路径缓存模型，避免每次交互都重新加载。"""
    return load_recognizer(checkpoint_path, DEVICE)


def image_to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def list_checkpoints() -> list[Path]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(
        [path for path in CHECKPOINT_DIR.glob("*.pt") if "_overfit" not in path.name],
        key=lambda path: path.name,
    )


st.title("PlateAlignNet")
st.caption("检测 → STN 几何矫正 → LPRNet 免分割识别")

available_checkpoints = list_checkpoints()

with st.sidebar:
    st.header("推理设置")
    backend_label = st.radio(
        "后端",
        ["真实模型（STN + LPRNet）", "演示启发式后端"],
        index=0 if available_checkpoints else 1,
    )
    use_real = backend_label.startswith("真实模型")

    checkpoint_path: Path | None = None
    box_source = "数据集真值框"
    manifest_path: Path | None = None
    manual_box: tuple[int, int, int, int] | None = None
    mode_label = "（由权重决定）"
    confidence = 0.25

    if use_real:
        if not available_checkpoints:
            st.error("outputs/ 下没有可用的 .pt 权重，先训练一个模型再回来。")
        else:
            chosen = st.selectbox("权重文件", [path.name for path in available_checkpoints])
            checkpoint_path = CHECKPOINT_DIR / chosen

        box_source = st.radio(
            "车牌框来源",
            ["数据集真值框", "手工指定矩形框"],
            help="YOLOv8 检测器本轮未训练。真值框模式只对数据集内的图片有效。",
        )
        if box_source == "数据集真值框":
            manifest_input = st.text_input("数据清单路径", value=str(DEFAULT_MANIFEST))
            manifest_path = Path(manifest_input)
            if not manifest_path.exists():
                st.warning("清单文件不存在，请先运行 ccpd_parser 生成。")
        st.slider("检测置信度（本模式未使用）", 0.05, 0.90, 0.25, 0.05, disabled=True)
    else:
        mode_label = st.selectbox("STN 模式", ["仿射（占位）", "透视（占位）", "关闭"])
        confidence = st.slider("检测置信度", 0.05, 0.90, 0.25, 0.05)

    st.divider()
    st.subheader("后续功能位置")
    st.checkbox("视频 / 摄像头", disabled=True)
    st.checkbox("批量识别", disabled=True)
    st.checkbox("无 STN / 仿射 / 透视并排消融", disabled=True)
    st.checkbox("历史记录与 CSV 导出", disabled=True)

mode_map = {
    "仿射（占位）": RectificationMode.AFFINE,
    "透视（占位）": RectificationMode.PERSPECTIVE,
    "关闭": RectificationMode.OFF,
}

real_info = None
real_model = None
if use_real and checkpoint_path is not None:
    try:
        real_model, real_info = get_real_model(str(checkpoint_path))
        st.markdown(
            f'<span class="real-badge">真实模型：{real_info.mode} · 参数 {real_info.total_parameters:,}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"STN {real_info.stn_parameters:,} + LPRNet {real_info.lprnet_parameters:,} 参数"
            f" · 权重 {Path(real_info.checkpoint).name}"
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"权重加载失败：{exc}")
        use_real = False
else:
    st.markdown(
        '<span class="demo-badge">当前为启发式演示后端，不代表模型精度</span>',
        unsafe_allow_html=True,
    )

demo_tab, architecture_tab, history_tab = st.tabs(["识别演示", "接口与扩展", "历史记录（预留）"])

with demo_tab:
    uploaded = st.file_uploader("上传车辆图片", type=["jpg", "jpeg", "png", "webp"])
    use_sample = st.toggle("使用内置样例", value=uploaded is None and not use_real)

    source_image = None
    image_name = None
    if uploaded is not None:
        try:
            source_image = Image.open(uploaded).convert("RGB")
            image_name = uploaded.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"无法读取图片：{exc}")
    elif use_sample:
        source_image = create_sample_image()

    if source_image is None:
        st.info("上传一张图片。真实模型请上传 CCPD 数据集内的图片（文件名需与清单一致）。")
    else:
        if use_real and box_source == "手工指定矩形框":
            width, height = source_image.size
            st.markdown("**手工框选车牌区域**")
            col_a, col_b, col_c, col_d = st.columns(4)
            x1 = col_a.number_input("x1", 0, width, int(width * 0.35), step=1)
            y1 = col_b.number_input("y1", 0, height, int(height * 0.60), step=1)
            x2 = col_c.number_input("x2", 0, width, int(width * 0.65), step=1)
            y2 = col_d.number_input("y2", 0, height, int(height * 0.80), step=1)
            manual_box = (int(x1), int(y1), int(x2), int(y2))

        options = InferenceOptions(
            rectification=RectificationMode.AFFINE if use_real else mode_map[mode_label],
            confidence_threshold=confidence,
            plate_size=PLATE_INPUT_SIZE,
        )

        response = None
        backend_name = "demo-heuristic"
        try:
            if use_real and real_model is not None:
                if manual_box is not None:
                    detector: object = ManualBoxDetector()
                    detector.set_box(manual_box)  # type: ignore[attr-defined]
                    backend_name = "real-model/manual-box"
                else:
                    detector = ManifestDetector(manifest_path)
                    backend_name = "real-model/ground-truth-box"
                pipeline = RealPlatePipeline(
                    detector=detector,  # type: ignore[arg-type]
                    rectifier=STNRectifier(real_model, DEVICE),  # type: ignore[arg-type]
                    recognizer=TorchRecognizer(real_model, DEVICE),  # type: ignore[arg-type]
                    backend_name=backend_name,
                )
                response = pipeline.run(source_image, options, image_name=image_name)
            else:
                response = get_demo_pipeline().run(source_image, options)
        except Exception as exc:  # noqa: BLE001
            st.error(f"推理失败：{exc}")

        if response is not None:
            left, right = st.columns([1.55, 1], gap="large")
            with left:
                st.subheader("检测结果")
                st.image(response.annotated_image or source_image, width="stretch")
                if use_real:
                    st.caption(f"框来源：{backend_name}（YOLOv8 检测器未训练，本页不展示检测精度）")
            with right:
                st.subheader("阶段结果")
                if response.status == InferenceStatus.ERROR:
                    st.error(response.message)
                elif response.status == InferenceStatus.NO_PLATE:
                    hint = (
                        "\n\n真值框模式下，图片文件名必须在数据清单里。可改用「手工指定矩形框」。"
                        if use_real
                        else ""
                    )
                    st.warning(response.message + hint)
                else:
                    plate = response.plates[0]
                    st.metric("车牌文本", plate.recognition.text or "（空）")
                    st.metric("识别置信度", f"{plate.recognition.confidence:.0%}")
                    st.metric("总耗时", f"{response.timing_ms['total']:.1f} ms")
                    before, after = st.columns(2)
                    before.image(plate.raw_plate, caption="原始车牌 ROI", width="stretch")
                    mode_text = real_info.mode if real_info else mode_label
                    after.image(
                        plate.rectified_plate,
                        caption=f"矫正输出 · {mode_text}",
                        width="stretch",
                    )
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
                    width="stretch",
                )
                download_json.download_button(
                    "下载结构化结果",
                    data=json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
                    file_name="platealignnet_result.json",
                    mime="application/json",
                    width="stretch",
                )

with architecture_tab:
    st.subheader("真实模型是怎么接进来的")
    st.code(
        """Detector.detect(image, confidence_threshold) -> list[Detection]
    真值框模式：按上传文件名在 samples.jsonl 里查出 bbox
    手工模式：界面输入框给出的矩形

Rectifier.rectify(plate_image, mode, output_size) -> image
    裁剪图 -> 188x48 -> 模型内的 STN -> 94x24 矫正图

Recognizer.recognize(plate_image) -> Recognition
    94x24 -> 模型内的 LPRNet -> CTC 解码 -> 文本 + 置信度

PlatePipeline.run(image, options) -> InferenceResponse""",
        language="python",
    )
    if real_info is not None:
        st.markdown("**当前权重自描述信息**")
        st.json(real_info.as_dict())
    st.markdown(
        "训练好的三种几何模式分别保存在 `outputs/end2end_off.pt`、`end2end_affine.pt`、"
        "`end2end_perspective.pt`，在左侧下拉切换即可对比。**几何模式由权重决定**，"
        "界面不会再改动模型内部的 STN。"
    )

with history_tab:
    st.info("此页保留给 SQLite/CSV 历史记录、筛选、错误案例画廊与实验统计。")
