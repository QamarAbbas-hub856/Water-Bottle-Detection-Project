import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip
import base64

from utils.history import save_history


# -----------------------------
# Custom Sub-Boxes Filter
# -----------------------------
def filter_sub_boxes(boxes):
    """
    Ek hi bottle par banne wale duplicate sub-boxes (caps/labels) ko remove karta hai.
    """
    if len(boxes) <= 1:
        return list(range(len(boxes)))

    xyxy = boxes.xyxy.cpu().numpy()
    keep = []

    heights = xyxy[:, 3] - xyxy[:, 1]
    order = np.argsort(heights)[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        box_i = xyxy[i]
        width_i = box_i[2] - box_i[0]

        remaining_indices = order[1:]
        suppress = []

        for idx, j in enumerate(remaining_indices):
            box_j = xyxy[j]

            x_left = max(box_i[0], box_j[0])
            x_right = min(box_i[2], box_j[2])
            horizontal_overlap = max(0, x_right - x_left)

            width_j = box_j[2] - box_j[0]
            min_width = min(width_i, width_j)

            if min_width > 0 and (horizontal_overlap / min_width) > 0.50:
                suppress.append(idx)

        order = np.delete(remaining_indices, suppress)

    return keep


# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-7/weights/best.pt")


# -----------------------------
# Helper Function
# -----------------------------
def display_custom_video(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    b64_video = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <div style="display:flex;justify-content:center;">
        <video width="100%" style="max-width:700px;border-radius:10px;" controls>
            <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
        </video>
    </div>
    """

    st.markdown(video_html, unsafe_allow_html=True)


# -----------------------------
# UI
# -----------------------------
st.title("🎥 Video Detection")
st.write("Upload a video and detect water bottles using YOLO11.")

st.markdown("---")

col_conf, col_iou = st.columns(2)

with col_conf:
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        key="video_conf"
    )

with col_iou:
    iou_thresh = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Lower value removes duplicate boxes on the same bottle",
        key="video_iou"
    )

uploaded_video = st.file_uploader(
    "📤 Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    t_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t_file.write(uploaded_video.read())
    t_file.close()

    st.subheader("📺 Original Video")
    display_custom_video(t_file.name)

    uploaded_video.seek(0)

    if st.button("🚀 Start Detection", use_container_width=True):

        model = load_model()

        with st.spinner("🎥 Processing Video..."):

            cap = cv2.VideoCapture(t_file.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            raw_output_path = "raw_output.mp4"
            final_output_path = "output_video.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)
            status = st.empty()

            frame_no = 0
            total_bottles = 0
            confidence_list = []

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(
                    frame,
                    conf=confidence,
                    iou=iou_thresh,
                    verbose=False
                )

                result = results[0]
                boxes = result.boxes

                if len(boxes) > 0:
                    keep_indices = filter_sub_boxes(boxes)
                    result.boxes = boxes[keep_indices]

                final_boxes = result.boxes
                bottle_count = len(final_boxes)

                total_bottles += bottle_count

                if bottle_count > 0:
                    confidence_list.extend(final_boxes.conf.cpu().numpy())

                annotated = result.plot(
                    line_width=3,
                    font_size=18
                )

                out.write(annotated)

                frame_no += 1

                progress.progress(
                    min(frame_no / max(total_frames, 1), 1.0)
                )

                status.write(
                    f"Processing Frame {frame_no}/{total_frames}"
                )

            cap.release()
            out.release()

            if os.path.exists(t_file.name):
                os.remove(t_file.name)

            status.write("🎬 Finalizing video format...")

            try:
                clip = VideoFileClip(raw_output_path)
                clip.write_videofile(
                    final_output_path,
                    codec="libx264",
                    audio=False,
                    logger=None
                )
                clip.close()

                if os.path.exists(raw_output_path):
                    os.remove(raw_output_path)

            except Exception:

                if os.path.exists(raw_output_path):
                    os.rename(raw_output_path, final_output_path)

            progress.empty()
            status.empty()

        avg_conf = (
            float(np.mean(confidence_list)) * 100
            if confidence_list else 0.0
        )

        save_history(
            "Video",
            total_bottles,
            avg_conf
        )

        st.success("✅ Video Detection Completed")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🍼 Bottles Detected", total_bottles)

        with c2:
            st.metric("🎯 Avg Confidence", f"{avg_conf:.2f}%")

        with c3:
            st.metric("🤖 Model", "YOLO11")

        st.subheader("📺 Detection Result Video")
        display_custom_video(final_output_path)

        with open(final_output_path, "rb") as file:

            st.download_button(
                label="⬇ Download Processed Video",
                data=file,
                file_name="detected_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

else:
    st.info("📤 Please upload a video.")