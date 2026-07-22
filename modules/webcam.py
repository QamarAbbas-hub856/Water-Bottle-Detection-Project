import streamlit as st
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

from utils.history import save_history


# -------------------------------------------------
# STUN Server Configuration
# -------------------------------------------------
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


# -------------------------------------------------
# Load YOLO Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-7/weights/best.pt")


model = load_model()


# -------------------------------------------------
# Remove Duplicate Sub Boxes
# -------------------------------------------------
def filter_sub_boxes(boxes):

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

        remaining = order[1:]
        suppress = []

        for idx, j in enumerate(remaining):

            box_j = xyxy[j]

            x_left = max(box_i[0], box_j[0])
            x_right = min(box_i[2], box_j[2])

            overlap = max(0, x_right - x_left)

            width_j = box_j[2] - box_j[0]

            min_width = min(width_i, width_j)

            if min_width > 0 and (overlap / min_width) > 0.60:
                suppress.append(idx)

        order = np.delete(remaining, suppress)

    return keep


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📷 Live Webcam Detection")

st.write("Detect water bottles in real time using YOLO11.")

st.markdown("---")


col1, col2 = st.columns(2)

with col1:

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        0.50,
        0.05,
        key="webcam_conf",
    )

with col2:

    iou_thresh = st.slider(
        "IoU (NMS) Threshold",
        0.10,
        1.00,
        0.45,
        0.05,
        help="Lower value removes duplicate boxes on same bottle.",
        key="webcam_iou",
    )


# -------------------------------------------------
# Video Processor
# -------------------------------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.history_saved = False

    def recv(self, frame):

        try:

            image = frame.to_ndarray(format="bgr24")

            results = model.predict(
                image,
                conf=confidence,
                iou=iou_thresh,
                verbose=False,
            )

            result = results[0]

            boxes = result.boxes

            if len(boxes) > 0:

                keep = filter_sub_boxes(boxes)

                result.boxes = boxes[keep]

            final_boxes = result.boxes

            bottle_count = len(final_boxes)

            if bottle_count > 0:

                avg_conf = (
                    float(np.mean(final_boxes.conf.cpu().numpy()))
                    * 100
                )

            else:

                avg_conf = 0.0

            if bottle_count > 0 and not self.history_saved:

                try:

                    save_history(
                        "Webcam",
                        bottle_count,
                        avg_conf,
                    )

                    self.history_saved = True

                except Exception:

                    pass

            if bottle_count == 0:

                self.history_saved = False

            annotated = result.plot(
                line_width=3,
                font_size=18,
            )

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )

        except Exception:

            return frame


# -------------------------------------------------
# Webcam Stream
# -------------------------------------------------
webrtc_streamer(
    key="water-bottle-webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)