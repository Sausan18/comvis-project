import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

st.set_page_config(layout="wide", page_title="Helmet Detection - CLAHE")
st.write("## Deteksi Helm menggunakan YOLOV8 dan CLAHE")
st.write(
    "Berikut merupakan visualisasi project computer vision mengenai Pendeteksian Pelanggaran Penggunaan Helm Berbasis YOLOV8 dan Peningkatan Citra CLAHE untuk Optimalisasi ETLE"
)

st.sidebar.write("## Pilih Video Streaming / Recording")

col1, col2 = st.columns(2)
table_placeholder1 = st.container()
table_placeholder2 = st.container()

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

add_selectbox = st.sidebar.selectbox(
    "Silahkan pilih CCTV yang akan ditampilkan",
    ("Stream 1 - Jambo Tape 1", "Stream 2 - Gelora 2", "Stream 3 - Ismud Gajah Mada",
     "Record 1", "Record 2", "Record 3", "Record 4", "Record 5", "Record 6")
)

clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 3.0, step=0.5)
tile_grid_size = st.sidebar.slider("CLAHE Tile Grid Size", 4, 16, 8, step=1)

STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1cc83753-c1ab-4b85-a5d7-e78a93f55fcc.m3u8"

if add_selectbox == "Stream 1 - Jambo Tape 1":
    STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1cc83753-c1ab-4b85-a5d7-e78a93f55fcc.m3u8"
elif add_selectbox == "Stream 2 - Gelora 2":
    STREAM_URL = "https://cctv.balitower.co.id/Gelora-017-700470_4/tracks-v1/index.fmp4.m3u8"
elif add_selectbox == "Stream 3 - Ismud Gajah Mada":
    STREAM_URL = "https://atcsdishub.pemkomedan.go.id/camera/ISMUDGAJAHMADA.m3u8"
elif add_selectbox.startswith("Record"):
    STREAM_URL = f"{add_selectbox.lower().replace(' ', '')}.mp4"

with col1:
    st.write("📽️ Tanpa CLAHE")
    frame_placeholder1 = st.empty()

with col2:
    st.write("📽️✨ Dengan CLAHE")
    frame_placeholder2 = st.empty()

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    st.error("Unable to open video stream.")
else:
    def extract_crops_and_counts(results, frame_rgb, class_helm=0):
        detections = results[0].boxes
        helm_crops, non_helm_crops = [], []
        helm_count, non_helm_count = 0, 0
        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame_rgb[y1:y2, x1:x2]
                if cls_id == class_helm:
                    helm_crops.append(crop)
                    helm_count += 1
                else:
                    non_helm_crops.append(crop)
                    non_helm_count += 1
        return helm_count, non_helm_count, helm_crops, non_helm_crops

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame from stream.")
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        rgb_frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v2 = clahe.apply(v)
        hsv = cv2.merge((h, s, v2))
        rgb_frame2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        results1 = model.predict(rgb_frame1, verbose=False)
        results2 = model.predict(rgb_frame2, verbose=False)

        annotated_frame1 = results1[0].plot()
        annotated_frame2 = results2[0].plot()

        pil_img1 = Image.fromarray(annotated_frame1)
        pil_img2 = Image.fromarray(annotated_frame2)
        frame_placeholder1.image(pil_img1, use_container_width=True)
        frame_placeholder2.image(pil_img2, use_container_width=True)

        count_helm1, count_nonhelm1, crops_helm1, crops_nonhelm1 = extract_crops_and_counts(results1, rgb_frame1)
        count_helm2, count_nonhelm2, crops_helm2, crops_nonhelm2 = extract_crops_and_counts(results2, rgb_frame2)

        with table_placeholder1:
            st.write("### 📊 Tabel Deteksi Tanpa CLAHE")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Helm: {count_helm1}**")
                for crop in crops_helm1[:2]:
                    st.image(crop, caption="Helm", width=120)
            with colB:
                st.markdown(f"**Non-Helm: {count_nonhelm1}**")
                for crop in crops_nonhelm1[:2]:
                    st.image(crop, caption="Non-Helm", width=120)

        with table_placeholder2:
            st.write("### 📊 Tabel Deteksi Dengan CLAHE")
            colC, colD = st.columns(2)
            with colC:
                st.markdown(f"**Helm: {count_helm2}**")
                for crop in crops_helm2[:2]:
                    st.image(crop, caption="Helm", width=120)
            with colD:
                st.markdown(f"**Non-Helm: {count_nonhelm2}**")
                for crop in crops_nonhelm2[:2]:
                    st.image(crop, caption="Non-Helm", width=120)

        time.sleep(0.03)  # stabilisasi

    cap.release()
