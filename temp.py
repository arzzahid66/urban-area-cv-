import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Urban Issue Detector", page_icon="🏙️", layout="centered")

st.title("🏙️ Urban Issue Detection")
st.markdown("Upload your trained model weight and an image to detect urban issues.")
st.divider()

# ── 1. Upload Weight File ──────────────────────────────────
st.subheader("1️⃣  Upload Model Weight (.pt)")
weight_file = st.file_uploader("Choose your .pt file", type=["pt"])

model = None
if weight_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(weight_file.read())
        weight_path = tmp.name
    try:
        from ultralytics import YOLO
        model = YOLO(weight_path)
        st.success(f"✅ Model loaded: **{weight_file.name}**")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

st.divider()

# ── 2. Upload Image ────────────────────────────────────────
st.subheader("2️⃣  Upload Image")
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# ── 3. Settings ────────────────────────────────────────────
conf = st.slider("Confidence Threshold", 0.10, 0.90, 0.30, 0.05)

# ── 4. Run Detection ───────────────────────────────────────
if model and image_file:
    image = Image.open(image_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(image, use_column_width=True)

    with st.spinner("Running detection..."):
        img_np = np.array(image)
        results = model.predict(img_np, conf=conf, verbose=False)
        annotated = results[0].plot()          # BGR numpy array with boxes drawn
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

    with col2:
        st.markdown("**Detections**")
        st.image(annotated_rgb, use_column_width=True)

    # Summary
    boxes = results[0].boxes
    if boxes and len(boxes):
        st.divider()
        st.subheader(f"📊 Found {len(boxes)} object(s)")
        names = results[0].names
        for box in boxes:
            cls  = names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            st.write(f"• **{cls}** — confidence: `{conf_val:.1%}`")
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

elif not model and image_file:
    st.warning("⬆️ Please upload a model weight file first.")
