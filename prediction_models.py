import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Issue Detector",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e2330;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Divider */
    hr { border-color: #2a2f3e; }

    /* Detection result rows */
    .det-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 14px;
        margin: 5px 0;
        background: #1e2330;
        border-left: 4px solid #00c9a7;
        border-radius: 6px;
        font-size: 0.95rem;
    }
    .det-row .cls  { font-weight: 700; color: #ffffff; }
    .det-row .conf { color: #00c9a7; font-weight: 600; }

    /* Status badges */
    .badge-ok  { background:#0d3b2e; color:#00c9a7; padding:2px 10px;
                 border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-err { background:#3b1515; color:#ff6b6b; padding:2px 10px;
                 border-radius:20px; font-size:0.8rem; font-weight:600; }

    /* Section headers */
    .section-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        margin: 18px 0 8px 0;
    }

    /* Image captions */
    .img-label {
        text-align: center;
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "🗑️  Trash Detection":      {"file": "weights/trash_best.pt",    "classes": "Glass · Metal · Paper · Plastic · Waste"},
    "✍️  Graffiti Detection":   {"file": "weights/graffiti_best.pt", "classes": "Graffiti"},
    "🚗  Illegal Parking":      {"file": "weights/parking_best.pt",  "classes": "Illegal parking"},
    "🕳️  Pothole Detection":    {"file": "weights/pothole_best.pt",  "classes": "Pothole"},
}

# ─────────────────────────────────────────────
# LOAD MODEL (CACHED)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Urban Detector")
    st.markdown("<div class='section-title'>Select Model</div>", unsafe_allow_html=True)

    # Radio with availability indicators
    model_labels = []
    for name, cfg in MODEL_REGISTRY.items():
        exists = os.path.exists(cfg["file"])
        status = "✅" if exists else "❌"
        model_labels.append(f"{status} {name}")

    selected_label = st.radio("", model_labels, label_visibility="collapsed")

    # Match label back to registry key safely (avoids emoji byte-length issues)
    selected_key = next(key for key in MODEL_REGISTRY if key in selected_label)
    selected_cfg = MODEL_REGISTRY[selected_key]

    st.divider()

    # Model info
    st.markdown("<div class='section-title'>Model Info</div>", unsafe_allow_html=True)
    st.markdown(f"**Detects:** {selected_cfg['classes']}")
    weight_exists = os.path.exists(selected_cfg["file"])
    if weight_exists:
        size_mb = os.path.getsize(selected_cfg["file"]) / 1e6
        st.markdown(f"**Weight:** `{selected_cfg['file'].split('/')[-1]}`")
        st.markdown(f"**Size:** `{size_mb:.1f} MB`")
        st.markdown("<span class='badge-ok'>READY</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Weight:** `{selected_cfg['file'].split('/')[-1]}`")
        st.markdown("<span class='badge-err'>MISSING</span>", unsafe_allow_html=True)
        st.caption(f"Place your `.pt` file at `{selected_cfg['file']}`")

    st.divider()

    # ── Confidence threshold ──────────────────
    st.markdown("<div class='section-title'>Detection Settings</div>", unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.95,
        value=0.30, step=0.05,
        help="Minimum confidence to show a detection. Lower = more detections, Higher = fewer but more certain."
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10, max_value=0.95,
        value=0.45, step=0.05,
        help="Controls overlap tolerance between boxes. Lower = fewer overlapping boxes."
    )

    # Quick preset buttons
    st.markdown("<div class='section-title'>Quick Presets</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    preset = None
    with col_a:
        if st.button("🔍 Low\n0.15", use_container_width=True):
            preset = 0.15
    with col_b:
        if st.button("⚖️ Mid\n0.30", use_container_width=True):
            preset = 0.30
    with col_c:
        if st.button("🎯 High\n0.60", use_container_width=True):
            preset = 0.60

    if preset:
        conf_threshold = preset
        st.caption(f"Preset applied: `{preset}`")

    st.divider()
    st.caption("YOLOv8 · Ultralytics · FYP 2024")

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("🏙️ Urban Issue Detection System")
st.caption("AI-powered detection of urban problems using YOLOv8 object detection")
st.divider()

# Stop if weights missing
if not weight_exists:
    st.error(
        f"**Model weight not found:** `{selected_cfg['file']}`  \n"
        "Create a `weights/` folder in your project directory and place the `.pt` file there."
    )
    st.stop()

# Load model
model = load_model(selected_cfg["file"])

# ── Upload ────────────────────────────────────
st.markdown("<div class='section-title'>Upload Image</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    # ── Run inference ─────────────────────────
    with st.spinner("Running detection..."):
        t0 = time.time()
        results = model.predict(
            source=img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        elapsed = (time.time() - t0) * 1000  # ms

    annotated_bgr = results[0].plot()
    annotated_rgb = Image.fromarray(annotated_bgr[..., ::-1])

    boxes   = results[0].boxes
    n_det   = len(boxes) if boxes is not None else 0

    # ── Top metrics row ───────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects Found",       str(n_det))
    m2.metric("Inference Time",      f"{elapsed:.0f} ms")
    m3.metric("Confidence Threshold", f"{conf_threshold:.0%}")
    m4.metric("IoU Threshold",        f"{iou_threshold:.0%}")

    st.divider()

    # ── Side-by-side images ───────────────────
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(image, use_column_width=True)
        st.markdown("<div class='img-label'>Original</div>", unsafe_allow_html=True)
    with img_col2:
        st.image(annotated_rgb, use_column_width=True)
        st.markdown("<div class='img-label'>Detections</div>", unsafe_allow_html=True)

    st.divider()

    # ── Detection results ─────────────────────
    st.markdown("<div class='section-title'>Detection Results</div>", unsafe_allow_html=True)

    if n_det == 0:
        st.warning("No objects detected. Try lowering the **Confidence Threshold** in the sidebar.")
    else:
        names = results[0].names

        # Class summary counts
        class_counts = {}
        for box in boxes:
            cls_name = names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # Summary chips
        chip_cols = st.columns(min(len(class_counts), 5))
        for i, (cls_name, count) in enumerate(class_counts.items()):
            with chip_cols[i % len(chip_cols)]:
                st.metric(cls_name, count)

        st.markdown("")

        # Per-detection rows sorted by confidence
        sorted_boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
        for i, box in enumerate(sorted_boxes):
            cls_name  = names[int(box.cls[0])]
            conf_val  = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1

            bar_pct = int(conf_val * 100)
            bar_fill = "#00c9a7" if conf_val >= 0.6 else "#f59e0b" if conf_val >= 0.35 else "#ef4444"

            st.markdown(f"""
            <div class='det-row'>
                <span class='cls'>#{i+1} &nbsp; {cls_name}</span>
                <span style='color:#9ca3af; font-size:0.82rem;'>{w}×{h}px</span>
                <span class='conf'>{conf_val:.1%}</span>
            </div>
            <div style='background:#1e2330; border-radius:0 0 6px 6px;
                        height:4px; margin-bottom:4px;'>
                <div style='width:{bar_pct}%; height:4px;
                            background:{bar_fill}; border-radius:4px;'></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Download annotated image ──────────
        st.divider()
        import io
        buf = io.BytesIO()
        annotated_rgb.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="📥 Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"detection_{selected_key.split()[1].lower()}.jpg",
            mime="image/jpeg"
        )

else:
    # ── Empty state ───────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#4b5563;'>
        <div style='font-size:3.5rem; margin-bottom:16px;'>📸</div>
        <div style='font-size:1.1rem; font-weight:600; color:#6b7280;'>
            Upload an image to start detection
        </div>
        <div style='font-size:0.85rem; margin-top:8px;'>
            Select a model from the sidebar, then upload a JPG or PNG image
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#374151; font-size:0.78rem;'>"
    "Urban Issue Detection System · YOLOv8 · FYP · Streamlit Cloud"
    "</div>",
    unsafe_allow_html=True
)