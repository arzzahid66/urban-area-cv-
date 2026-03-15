import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import torch

st.set_page_config(page_title="Trash Detection", page_icon="🗑️", layout="centered")

st.title("🗑️ Trash Detection App")
st.markdown("Upload your **YOLOv8 model** and an **image** to detect trash categories.")
st.markdown("**Classes:** `Glass` · `Metal` · `Paper` · `Plastic` · `Waste`")
st.divider()

CLASS_COLORS = {
    "glass":   (59,  130, 246),
    "metal":   (107, 114, 128),
    "paper":   (245, 158,  11),
    "plastic": ( 16, 185, 129),
    "waste":   (239,  68,  68),
}

def draw_boxes(image: Image.Image, boxes, scores, labels, names) -> Image.Image:
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        cls_name = names[int(label)].lower()
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_name, (139, 92, 246))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        lbl_text = f"{cls_name.capitalize()} {score:.0%}"
        bbox = draw.textbbox((x1, max(0, y1 - 18)), lbl_text)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, max(0, y1 - 18)), lbl_text, fill="white")
    return img

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_file  = st.file_uploader("📦 Upload model (.pt)", type=["pt"])
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)
    st.divider()
    st.caption("Model: YOLOv8s · FYP Urban Issue Detection")

@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    # Use torch directly — no ultralytics, no opencv dependency
    model = torch.hub.load(
        "ultralytics/yolov5", "custom",
        path=path, force_reload=False, verbose=False,
        trust_repo=True
    )
    model.eval()
    return model

# Try ultralytics YOLO as fallback (some .pt files need it)
@st.cache_resource(show_spinner="Loading model…")
def load_ultralytics_model(path: str):
    # Patch: delete Qt platform plugins so libGL is never loaded
    import shutil, pathlib
    qt_plugins = pathlib.Path("/usr/local/lib/python3.12/dist-packages/cv2/qt")
    if qt_plugins.exists():
        shutil.rmtree(str(qt_plugins), ignore_errors=True)
    opencv_python_libs = pathlib.Path("/usr/local/lib/python3.12/dist-packages/opencv_python.libs")
    if opencv_python_libs.exists():
        for f in opencv_python_libs.glob("libQt5*"):
            f.unlink(missing_ok=True)
    from ultralytics import YOLO
    return YOLO(path)

model      = None
model_type = None

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        model      = load_ultralytics_model(tmp_path)
        model_type = "ultralytics"
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model:\n{e}")
else:
    st.info("👈 Upload your `trash_best.pt` model in the sidebar to get started.")

# ── Inference ─────────────────────────────────────────────────
image_file = st.file_uploader("🖼️ Upload a trash image", type=["jpg","jpeg","png","webp","bmp"])

if image_file and model:
    image = Image.open(image_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection…"):
        img_np  = np.array(image)
        results = model.predict(source=img_np, conf=conf_thresh, iou=iou_thresh, verbose=False)
        result  = results[0]

        raw_boxes  = result.boxes.xyxy.cpu().numpy()
        raw_scores = result.boxes.conf.cpu().numpy()
        raw_labels = result.boxes.cls.cpu().numpy()
        names      = model.names

        plotted_img = draw_boxes(image, raw_boxes, raw_scores, raw_labels, names)

    with col2:
        st.subheader("Detected Objects")
        st.image(plotted_img, use_container_width=True)

    st.divider()

    if len(raw_boxes) == 0:
        st.warning("⚠️ No trash detected. Try lowering the confidence threshold.")
    else:
        st.subheader(f"📋 Detection Summary — {len(raw_boxes)} object(s) found")

        class_counts: dict[str, int] = {}
        detections = []
        for score, label in zip(raw_scores, raw_labels):
            cls_name = names[int(label)].capitalize()
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            detections.append({"Class": cls_name, "Confidence": f"{score:.1%}"})

        badge_cols = st.columns(len(class_counts))
        for idx, (cls_name, count) in enumerate(sorted(class_counts.items())):
            r, g, b = CLASS_COLORS.get(cls_name.lower(), (139, 92, 246))
            with badge_cols[idx]:
                st.markdown(
                    f"""<div style="background:rgb({r},{g},{b});padding:10px 6px;
                        border-radius:10px;text-align:center;color:white;font-weight:600;">
                        {cls_name}<br><span style="font-size:1.4rem">{count}</span></div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.dataframe(detections, use_container_width=True, hide_index=True)

elif image_file and not model:
    st.warning("⚠️ Please upload a model first (sidebar).")
