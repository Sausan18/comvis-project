import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

time.sleep(0.02)

st.set_page_config(layout="wide", page_title="Helmet Detection - CLAHE")
st.write("## Deteksi Helm menggunakan YOLOV8 dan CLAHE")
st.write(
    "Berikut merupakan visualisasi project computer vision mengenai Pendeteksian Pelanggaran Penggunaan Helm Berbasis YOLOV8 dan Peningkatan Citra CLAHE untuk Optimalisasi ETLE"
)
st.sidebar.write("## Pilih Video Streaming / Recording")


header = st.container()
# model-run = st.container()
col1, col2 = st.columns(2)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Load your YOLO model
# model = YOLO("best.pt")  # or "yolov8n.pt" if testing
model = load_model()

# Using object notation

add_selectbox = st.sidebar.selectbox(
    "Silahkan pilih CCTV yang akan ditampilkan",
    ("Stream 1 - Jambo Tape 1", "Stream 2 - Gelora 2", "Stream 3 - Ismud Gajah Mada", "Record 1", "Record 2", "Record 3", "Record 4", "Record 5", "Record 6")
)

# Sidebar sliders for CLAHE parameters
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 3.0, step=0.5)
tile_grid_size = st.sidebar.slider("CLAHE Tile Grid Size", 4, 16, 8, step=1)

STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1cc83753-c1ab-4b85-a5d7-e78a93f55fcc.m3u8"

# .m3u8 stream URL
if add_selectbox == "Stream 1 - Jambo Tape 1":
    STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1cc83753-c1ab-4b85-a5d7-e78a93f55fcc.m3u8"
elif add_selectbox == "Stream 2 - Gelora 2":
    STREAM_URL = "https://cctv.balitower.co.id/Gelora-017-700470_4/tracks-v1/index.fmp4.m3u8"
elif add_selectbox == "Stream 3 - Ismud Gajah Mada":
    STREAM_URL = "https://atcsdishub.pemkomedan.go.id/camera/ISMUDGAJAHMADA.m3u8"
elif add_selectbox == "Record 1":
    STREAM_URL = "record12.mp4"
elif add_selectbox == "Record 2":
    STREAM_URL = "record2.mp4"
elif add_selectbox == "Record 3":
    STREAM_URL = "record3.mp4"
elif add_selectbox == "Record 4":
    STREAM_URL = "record4.mp4"
elif add_selectbox == "Record 5":
    STREAM_URL = "record5.mp4"
elif add_selectbox == "Record 6":
    STREAM_URL = "record6.mp4"

# with header:
#     st.title("Deteksi Pengguna Helm dengan YOLOV8 dan CLAHE pada ETLE")

# with model-run:
with col1:
    st.write("📽️ Tanpa CLAHE")
    # Create a video container
    frame_placeholder1 = st.empty()

with col2:
    st.write("📽️✨ Dengan CLAHE")
    # Create a video container
    frame_placeholder2 = st.empty()

# Open the video stream
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    st.error("Unable to open video stream.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        frame_count += 1
        if frame_count % 3 != 0:
            continue  

        if not ret:
            st.warning("Failed to read frame from stream.")
            break

        # # Rotate only for 'record' videos
        # if "record" in STREAM_URL.lower():
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert BGR to RGB
        rgb_frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply CLAHE
        # clahe = cv2.createCLAHE(clipLimit=3.,tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        v2 = clahe.apply(v)
        hsv = cv2.merge((h,s,v2))
        rgb_frame2 = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

        # Apply YOLO model
        results1 = model.predict(rgb_frame1, verbose=False)
        results2 = model.predict(rgb_frame2, verbose=False)

        # Plot results on frame
        annotated_frame1 = results1[0].plot()
        annotated_frame2 = results2[0].plot()

        # Convert to PIL image and display
        pil_img1 = Image.fromarray(annotated_frame1)
        pil_img2 = Image.fromarray(annotated_frame2)
        frame_placeholder1.image(pil_img1, use_column_width=True)
        frame_placeholder2.image(pil_img2, use_column_width=True)

cap.release()
