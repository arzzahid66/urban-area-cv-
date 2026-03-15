import os
# ── Force headless OpenCV BEFORE any cv2/ultralytics import ──
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Trash Detection",
    page_icon="🗑️",
    layout="centered",
)

# ── Title ─────────────────────────────────────────────────────
st.title("🗑️ Trash Detection App")
st.markdown("Upload your **YOLOv8 model** and an **image** to detect trash categories.")
st.markdown("**Classes:** `Glass` · `Metal` · `Paper` · `Plastic` · `Waste`")
st.divider()

# ── Class colours ─────────────────────────────────────────────
CLASS_COLORS = {
    "glass":   "#3B82F6",
    "metal":   "#6B7280",
    "paper":   "#F59E0B",
    "plastic": "#10B981",
    "waste":   "#EF4444",
}

# PIL hex → RGB tuple
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ── Draw boxes using Pillow (no cv2 needed at all) ────────────
def draw_boxes(image: Image.Image, result, names: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    boxes = result.boxes
    if boxes is None:
        return img
    for box in boxes:
        cls_id   = int(box.cls[0])
        cls_name = names[cls_id].capitalize()
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = hex_to_rgb(CLASS_COLORS.get(cls_name.lower(), "#8B5CF6"))
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Label background
        label = f"{cls_name} {conf_val:.0%}"
        bbox = draw.textbbox((x1, y1), label)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1), label, fill="white")
    return img

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_file  = st.file_uploader("📦 Upload model (.pt)", type=["pt"])
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)
    st.divider()
    st.caption("Model: YOLOv8s · FYP Urban Issue Detection")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    # Import here so env vars are set first
    from ultralytics import YOLO
    return YOLO(path)

model = None
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        model = load_model(tmp_path)
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model:\n{e}")
else:
    st.info("👈 Upload your `trash_best.pt` model in the sidebar to get started.")

# ── Image upload & inference ──────────────────────────────────
image_file = st.file_uploader("🖼️ Upload a trash image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if image_file and model:
    image = Image.open(image_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection…"):
        img_array = np.array(image)
        results = model.predict(
            source=img_array,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )
        result      = results[0]
        plotted_img = draw_boxes(image, result, model.names)  # pure Pillow, no cv2

    with col2:
        st.subheader("Detected Objects")
        st.image(plotted_img, use_container_width=True)

    # ── Detection summary ─────────────────────────────────────
    st.divider()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        st.warning("⚠️ No trash detected. Try lowering the confidence threshold.")
    else:
        st.subheader(f"📋 Detection Summary  —  {len(boxes)} object(s) found")

        names = model.names
        class_counts: dict[str, int] = {}
        detections = []

        for box in boxes:
            cls_id   = int(box.cls[0])
            cls_name = names[cls_id].capitalize()
            conf_val = float(box.conf[0])
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            detections.append({"Class": cls_name, "Confidence": f"{conf_val:.1%}"})

        badge_cols = st.columns(len(class_counts))
        for idx, (cls_name, count) in enumerate(sorted(class_counts.items())):
            color = CLASS_COLORS.get(cls_name.lower(), "#8B5CF6")
            with badge_cols[idx]:
                st.markdown(
                    f"""
                    <div style="background:{color};padding:10px 6px;border-radius:10px;
                                text-align:center;color:white;font-weight:600;">
                        {cls_name}<br><span style="font-size:1.4rem">{count}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.dataframe(detections, use_container_width=True, hide_index=True)

elif image_file and not model:
    st.warning("⚠️ Please upload a model first (sidebar).")
