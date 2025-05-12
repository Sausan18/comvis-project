import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import base64
from io import BytesIO
from datetime import datetime

st.set_page_config(layout="wide", page_title="Helmet Detection - CLAHE")
st.title("Deteksi Helm menggunakan YOLOV8 dan CLAHE")
st.sidebar.title("Konfigurasi")
st.sidebar.write("## Pilih Video Streaming / Recording")

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

STREAM_URL = ""
if add_selectbox == "Stream 1 - Jambo Tape 1":
    STREAM_URL = "https://cctv-stream.bandaacehkota.info/memfs/1cc83753-c1ab-4b85-a5d7-e78a93f55fcc.m3u8"
elif add_selectbox == "Stream 2 - Gelora 2":
    STREAM_URL = "https://cctv.balitower.co.id/Gelora-017-700470_4/tracks-v1/index.fmp4.m3u8"
elif add_selectbox == "Stream 3 - Ismud Gajah Mada":
    STREAM_URL = "https://atcsdishub.pemkomedan.go.id/camera/ISMUDGAJAHMADA.m3u8"
else:
    STREAM_URL = f"record{add_selectbox[-1]}.mp4"

col1, col2 = st.columns(2)
frame_placeholder1 = col1.empty()
frame_placeholder2 = col2.empty()

if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

def encode_image_to_base64(img_array):
    if img_array is None or img_array.size == 0:
        return ""
    img_pil = Image.fromarray(img_array).resize((64, 64))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def extract_crops_and_counts(results, frame_rgb):
    detections = results[0].boxes
    helm_crops, non_helm_crops = [], []
    helm_count, non_helm_count = 0, 0

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame_rgb[y1:y2, x1:x2]
            if cls_id == 0:
                helm_crops.append(crop)
                helm_count += 1
            elif cls_id == 1:
                non_helm_crops.append(crop)
                non_helm_count += 1

    return helm_count, non_helm_count, helm_crops, non_helm_crops

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    st.error("Tidak dapat membuka stream video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        rgb_frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))
        rgb_frame2 = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

        results1 = model.predict(rgb_frame1, verbose=False)
        results2 = model.predict(rgb_frame2, verbose=False)

        annotated_frame1 = results1[0].plot()
        annotated_frame2 = results2[0].plot()

        frame_placeholder1.image(Image.fromarray(annotated_frame1), use_container_width=True, caption="Tanpa CLAHE")
        frame_placeholder2.image(Image.fromarray(annotated_frame2), use_container_width=True, caption="Dengan CLAHE")

        count_helm1, count_nonhelm1, crops_helm1, crops_nonhelm1 = extract_crops_and_counts(results1, rgb_frame1)
        count_helm2, count_nonhelm2, crops_helm2, crops_nonhelm2 = extract_crops_and_counts(results2, rgb_frame2)

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            'timestamp': timestamp,
            'helm1': count_helm1,
            'nonhelm1': count_nonhelm1,
            'img1_helm': crops_helm1[0] if crops_helm1 else None,
            'img1_nonhelm': crops_nonhelm1[0] if crops_nonhelm1 else None,
            'helm2': count_helm2,
            'nonhelm2': count_nonhelm2,
            'img2_helm': crops_helm2[0] if crops_helm2 else None,
            'img2_nonhelm': crops_nonhelm2[0] if crops_nonhelm2 else None
        }
        st.session_state.detection_log.append(entry)

        if len(st.session_state.detection_log) >= 10:
            break

    cap.release()

    st.markdown("""
    <style>
    .custom-table {
        border-collapse: collapse;
        width: 100%;
    }
    .custom-table th, .custom-table td {
        border: 1px solid #ddd;
        text-align: center;
        padding: 6px;
        font-size: 14px;
    }
    .custom-table th {
        background-color: #f2f2f2;
    }
    </style>
    """, unsafe_allow_html=True)

    html_table = """
    <table class="custom-table">
        <thead>
            <tr>
                <th rowspan="2">Timestamp</th>
                <th colspan="2">Tanpa CLAHE</th>
                <th colspan="2">Dengan CLAHE</th>
            </tr>
            <tr>
                <th>Jumlah</th>
                <th>Gambar</th>
                <th>Jumlah</th>
                <th>Gambar</th>
            </tr>
        </thead>
        <tbody>
    """

    for log in st.session_state.detection_log[-10:][::-1]:
        img1 = encode_image_to_base64(log["img1_helm"] or log["img1_nonhelm"])
        img2 = encode_image_to_base64(log["img2_helm"] or log["img2_nonhelm"])

        html_table += f"""
            <tr>
                <td>{log['timestamp']}</td>
                <td>{log['helm1'] + log['nonhelm1']}</td>
                <td><img src="{img1}" width="64"/></td>
                <td>{log['helm2'] + log['nonhelm2']}</td>
                <td><img src="{img2}" width="64"/></td>
            </tr>
        """

    html_table += "</tbody></table>"
    st.markdown("### 🧾 Rekap Deteksi Helm dan Pelanggaran")
    st.markdown(html_table, unsafe_allow_html=True)
