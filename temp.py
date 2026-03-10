"""
============================================================
AI-Based Urban Issue Detection System
Universal Streamlit App — FYP
============================================================
Upload your 4 trained .pt weight files to weights/ folder
then run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Urban Issue Detection System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL REGISTRY
# ============================================================
MODEL_REGISTRY = {
    "🗑️ Trash Detection": {
        "weight_file": "weights/trash_best.pt",
        "classes": ["Glass", "Metal", "Paper", "Plastic", "Waste"],
        "description": "Detects trash and litter types on streets and public spaces",
        "colors": {
            "Glass":   (255, 100, 100),
            "Metal":   (100, 255, 100),
            "Paper":   (100, 100, 255),
            "Plastic": (255, 255, 100),
            "Waste":   (255, 100, 255),
        },
        "conf_default": 0.30,
        "icon": "🗑️"
    },
    "✍️ Graffiti Detection": {
        "weight_file": "weights/graffiti_best.pt",
        "classes": ["graffiti"],
        "description": "Detects graffiti vandalism on walls and public property",
        "colors": {
            "graffiti": (255, 80, 80),
            "0":        (255, 80, 80),
        },
        "conf_default": 0.30,
        "icon": "✍️"
    },
    "🚗 Illegal Parking Detection": {
        "weight_file": "weights/parking_best.pt",
        "classes": ["illegal"],
        "description": "Detects vehicles parked in restricted/illegal zones",
        "colors": {
            "illegal": (255, 50, 50),
        },
        "conf_default": 0.35,
        "icon": "🚗"
    },
    "🕳️ Pothole Detection": {
        "weight_file": "weights/pothole_best.pt",
        "classes": ["pothole"],
        "description": "Detects road potholes causing damage and safety hazards",
        "colors": {
            "pothole": (200, 100, 255),
        },
        "conf_default": 0.30,
        "icon": "🕳️"
    },
}

# ============================================================
# LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching — only loads once per session."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def check_weights_available():
    """Check which weight files exist."""
    available = {}
    for name, cfg in MODEL_REGISTRY.items():
        available[name] = os.path.exists(cfg["weight_file"])
    return available


def run_inference(model, image_np, conf_threshold, model_config):
    """Run YOLO inference and draw bounding boxes."""
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        verbose=False,
        device='cpu'  # EC2 CPU inference
    )

    result = results[0]
    annotated = image_np.copy()
    detections = []

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Get class name
            if hasattr(result, 'names') and cls_id in result.names:
                cls_name = result.names[cls_id]
            elif cls_id < len(model_config["classes"]):
                cls_name = model_config["classes"][cls_id]
            else:
                cls_name = f"class_{cls_id}"

            # Get color
            colors = model_config["colors"]
            color = colors.get(cls_name, colors.get(list(colors.keys())[0], (0, 255, 0)))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return annotated, detections


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/city.png", width=80)
    st.title("Urban Detection")
    st.markdown("**AI-Based Urban Issue Detection System**")
    st.markdown("*Final Year Project — Computer Vision*")
    st.divider()

    # Model selector
    st.markdown("### 🤖 Select Model")
    available = check_weights_available()

    model_options = []
    for name in MODEL_REGISTRY.keys():
        status = "✅" if available[name] else "❌"
        model_options.append(f"{status} {name}")

    selected_display = st.selectbox(
        "Detection Type",
        options=model_options,
        index=0
    )

    # Strip status prefix to get actual key
    selected_model_key = selected_display[3:]  # Remove "✅ " or "❌ "
    model_config = MODEL_REGISTRY[selected_model_key]

    st.divider()

    # Confidence slider
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=model_config["conf_default"],
        step=0.05,
        help="Higher = fewer but more confident detections"
    )

    st.divider()

    # Info panel
    st.markdown("### ℹ️ Model Info")
    st.markdown(f"**Description:**")
    st.info(model_config["description"])
    st.markdown(f"**Classes:** {', '.join(model_config['classes'])}")
    st.markdown(f"**Weight file:** `{model_config['weight_file']}`")

    # Weight file status
    if available[selected_model_key]:
        st.success("✅ Weight file found!")
    else:
        st.error("❌ Weight file missing!")
        st.markdown(f"""
        **To fix:**
        1. Train model with the corresponding Colab notebook
        2. Download `best.pt` from Google Drive
        3. Upload to your EC2 server:
        ```
        scp -i key.pem best.pt \\
          ubuntu@EC2_IP:~/urban_detection/weights/{model_config['weight_file'].split('/')[-1]}
        ```
        """)

    st.divider()
    st.markdown("### 📊 All Models Status")
    for name, cfg in MODEL_REGISTRY.items():
        icon = cfg["icon"]
        status = "✅" if available[name] else "❌"
        short_name = name.split(" ", 1)[1]
        st.markdown(f"{status} {icon} {short_name}")


# ============================================================
# MAIN AREA
# ============================================================
st.title("🏙️ AI-Based Urban Issue Detection System")
st.markdown("Upload an image to detect urban problems using trained YOLOv8 models.")

# Active model banner
if available[selected_model_key]:
    st.success(f"**Active Model:** {selected_model_key}  |  **Confidence:** {conf_threshold:.0%}")
else:
    st.error(f"**Active Model:** {selected_model_key}  |  ⚠️ Weight file not found — see sidebar for upload instructions")

st.divider()

# ============================================================
# INPUT SECTION
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported: JPG, PNG, BMP, WEBP"
    )

    use_sample = st.checkbox("Use sample test image (URL)")
    sample_url = ""
    if use_sample:
        sample_url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg"
        )

with col2:
    st.markdown("### 💡 Tips for Best Results")
    tips = {
        "🗑️ Trash Detection":       "Works best on street-level photos. Ensure trash is visible and not obstructed.",
        "✍️ Graffiti Detection":     "Clear photos of walls. Good contrast helps. Works on various surfaces.",
        "🚗 Illegal Parking Detection": "Overhead or slight angle works best. Ensure vehicles are clearly visible.",
        "🕳️ Pothole Detection":      "Road-level dashcam angle works best. Ensure road surface is clearly lit.",
    }
    st.info(tips.get(selected_model_key, "Upload a clear image for best results."))

    st.markdown("**Recommended image specs:**")
    st.markdown("- Resolution: 640×640 or higher")
    st.markdown("- Format: JPG or PNG")
    st.markdown("- Good lighting, minimal blur")

st.divider()

# ============================================================
# RUN INFERENCE
# ============================================================
if uploaded_file is not None or (use_sample and sample_url):

    # Load image
    try:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            import requests
            response = requests.get(sample_url, timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()

    # Show original
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("### 📸 Original Image")
        st.image(image, use_column_width=True)
        st.caption(f"Size: {image.width} × {image.height} px")

    # Run inference
    if not available[selected_model_key]:
        with res_col2:
            st.markdown("### ⚠️ Cannot Run Inference")
            st.error(f"Weight file `{model_config['weight_file']}` not found.\n\nPlease upload the trained model weights to the server.")
    else:
        weight_path = model_config["weight_file"]

        with st.spinner(f"🔍 Running {selected_model_key}..."):
            model, error = load_model(weight_path)

            if error:
                st.error(f"Failed to load model: {error}")
                st.stop()

            start_time = time.time()
            annotated_bgr, detections = run_inference(
                model, image_bgr, conf_threshold, model_config
            )
            inference_time = time.time() - start_time

        # Convert back to RGB for display
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with res_col2:
            st.markdown("### 🎯 Detection Results")
            st.image(annotated_rgb, use_column_width=True)
            st.caption(f"Inference time: {inference_time*1000:.0f}ms  |  Found: {len(detections)} object(s)")

        # ============================================================
        # RESULTS SUMMARY
        # ============================================================
        st.divider()
        st.markdown("### 📊 Detection Summary")

        if len(detections) == 0:
            st.warning("No objects detected. Try lowering the confidence threshold or use a clearer image.")
        else:
            # Aggregate by class
            class_counts = {}
            class_confs = {}
            for det in detections:
                cls = det["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
                class_confs.setdefault(cls, []).append(det["confidence"])

            # Summary cards
            num_cols = min(len(class_counts), 4)
            cols = st.columns(num_cols)
            for i, (cls_name, count) in enumerate(class_counts.items()):
                avg_conf = np.mean(class_confs[cls_name])
                with cols[i % num_cols]:
                    st.metric(
                        label=cls_name,
                        value=f"{count} detected",
                        delta=f"Avg conf: {avg_conf:.1%}"
                    )

            # Detail table
            st.markdown("#### 📋 Detection Details")
            import pandas as pd
            df_data = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                df_data.append({
                    "#": i + 1,
                    "Class": det["class"],
                    "Confidence": f"{det['confidence']:.1%}",
                    "BBox (x1,y1,x2,y2)": f"({x1},{y1}) → ({x2},{y2})",
                    "Width×Height": f"{x2-x1}×{y2-y1}px"
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download annotated image
            st.divider()
            st.markdown("### 💾 Download Results")
            annotated_pil = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buf,
                    file_name=f"urban_detection_{selected_model_key.split()[1].lower()}.jpg",
                    mime="image/jpeg"
                )
            with col_dl2:
                # Download JSON report
                import json
                report = {
                    "model": selected_model_key,
                    "confidence_threshold": conf_threshold,
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "total_detections": len(detections),
                    "detections": detections
                }
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name="detection_report.json",
                    mime="application/json"
                )

else:
    # No image — show demo placeholder
    st.markdown("### 👆 Upload an image to get started")
    st.markdown("")

    demo_cols = st.columns(4)
    demo_cards = [
        ("🗑️", "Trash Detection", "Identifies Glass, Metal, Paper, Plastic & Waste", "6,000 training images"),
        ("✍️", "Graffiti Detection", "Identifies graffiti vandalism on surfaces", "2,148 training images"),
        ("🚗", "Illegal Parking", "Flags vehicles in restricted zones", "774 training images"),
        ("🕳️", "Pothole Detection", "Detects road damage and potholes", "1,390 training images"),
    ]
    for col, (icon, title, desc, data) in zip(demo_cols, demo_cards):
        with col:
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background: #f8f9fa;
                min-height: 180px;
            ">
                <div style="font-size: 2.5rem;">{icon}</div>
                <h4 style="margin: 10px 0 5px 0; color: #333;">{title}</h4>
                <p style="font-size: 0.85rem; color: #666; margin: 0;">{desc}</p>
                <p style="font-size: 0.75rem; color: #999; margin-top: 8px;">📊 {data}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    🏙️ AI-Based Urban Issue Detection System &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp;
    Built with YOLOv8 + Streamlit &nbsp;|&nbsp; Trained on Google Colab T4 GPU
</div>
""", unsafe_allow_html=True)