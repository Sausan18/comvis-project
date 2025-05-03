import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Load your YOLO model
# model = YOLO("best.pt")  # or "yolov8n.pt" if testing
model = load_model()

# Using object notation

add_selectbox = st.sidebar.selectbox(
    "Silahkan pilih CCTV yang akan ditampilkan",
    ("Banda Aceh - Simpang Dharma 3", "Jakarta - Bendungan Hilir 3", "Medan - Ismud Gajah Mada")
)

STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1e560ac1-8b57-416a-b64e-d4190ff83f88_output_0.m3u8"

# .m3u8 stream URL
if add_selectbox == "Simpang Dharma 3":
    STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1e560ac1-8b57-416a-b64e-d4190ff83f88_output_0.m3u8"
elif add_selectbox == "Jakarta - Bendungan Hilir 3":
    STREAM_URL = "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_2/tracks-v1/index.fmp4.m3u8"
elif add_selectbox == "Medan - Ismud Gajah Mada":
    STREAM_URL = "https://atcsdishub.pemkomedan.go.id/camera/ISMUDGAJAHMADA.m3u8"

st.title("Deteksi Pengguna Helm dengan YOLOV8 dan CLAHE pada ETLE")

# Create a video container
frame_placeholder = st.empty()

# Open the video stream
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    st.error("Unable to open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from stream.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply YOLO model
        results = model.predict(rgb_frame, verbose=False)

        # Plot results on frame
        annotated_frame = results[0].plot()

        # Convert to PIL image and display
        pil_img = Image.fromarray(annotated_frame)
        frame_placeholder.image(pil_img)

cap.release()
