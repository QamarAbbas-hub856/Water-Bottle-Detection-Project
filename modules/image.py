import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np

from utils.detector import detect_image
from utils.history import save_history


st.title("🖼 Image Detection")
st.write("Upload an image and detect water bottles using your trained YOLO model.")

st.markdown("---")

# ==========================================
# Detection Settings (Confidence & IoU)
# ==========================================

col_conf, col_iou = st.columns(2)

with col_conf:
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        key="img_conf"
    )

with col_iou:
    iou_thresh = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Lower value removes duplicate boxes on the same bottle",
        key="img_iou"
    )

uploaded = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Start Detection", use_container_width=True):

        with st.spinner("🔍 Detecting bottles..."):

            try:

                annotated, bottle_count, avg_conf = detect_image(
                    np.array(image),
                    confidence,
                    iou_thresh
                )

            except Exception as e:
                st.error(f"Detection Error:\n\n{e}")
                st.stop()

        with col2:

            st.subheader("Detection Result")

            st.image(
                annotated,
                use_container_width=True
            )

        # ==========================================
        # Save Detection History
        # ==========================================

        try:
            save_history(
                mode="Image",
                bottle_count=bottle_count,
                avg_conf=avg_conf
            )
        except Exception:
            pass

        # ==========================================
        # Result Message
        # ==========================================

        if bottle_count > 0:
            st.success("✅ Detection Completed Successfully!")
        else:
            st.warning("⚠ No Bottle Detected.")

        # ==========================================
        # Metrics
        # ==========================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🍼 Bottles", bottle_count)

        with c2:
            st.metric("🎯 Avg Confidence", f"{avg_conf:.2f}%")

        with c3:
            st.metric("🤖 Model", "YOLO11")

        st.markdown("---")

        # ==========================================
        # Download Result (Without OpenCV)
        # ==========================================

        result_image = Image.fromarray(annotated)

        buffer = BytesIO()
        result_image.save(buffer, format="JPEG")
        buffer.seek(0)

        st.download_button(
            label="⬇ Download Result",
            data=buffer,
            file_name="detected_image.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

else:

    st.info("📤 Please upload an image to begin detection.")