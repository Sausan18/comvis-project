import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ==================== Konfigurasi Halaman ==================== #
st.set_page_config(layout="wide", page_title="Helmet Detection - CLAHE")

# ==================== Header Aplikasi ==================== #
def show_header():
    st.title("Deteksi Helm menggunakan YOLOV8 dan CLAHE")
    st.markdown(
        "Berikut merupakan visualisasi project computer vision mengenai "
        "**Pendeteksian Pelanggaran Penggunaan Helm Berbasis YOLOV8 dan Peningkatan Citra CLAHE untuk Optimalisasi ETLE**"
    )

# ==================== Sidebar ==================== #
def sidebar_controls():
    selected_record = st.sidebar.selectbox(
        "Silahkan pilih rekaman video yang akan ditampilkan",
        ["Record 1", "Record 2", "Record 3", "Record 4", "Record 5", "Record 6"]
    )

    clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 3.0, step=0.5)
    tile_grid_size = st.sidebar.slider("CLAHE Tile Grid Size", 4, 16, 8, step=1)

    record_paths = {
        "Record 1": "video1.mp4",
        "Record 2": "record2.mp4",
        "Record 3": "record3.mp4",
        "Record 4": "record4.mp4",
        "Record 5": "record5.mp4",
        "Record 6": "record6.mp4"
    }

    return record_paths[selected_record], clip_limit, tile_grid_size

# ==================== Load YOLO Model ==================== #
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# ==================== Apply CLAHE ==================== #
def apply_clahe(frame, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge((h, s, v_clahe))
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

# ==================== Proses Frame Deteksi ==================== #
def process_frame(frame, model, clip_limit, tile_grid_size):
    rgb_no_clahe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_with_clahe = apply_clahe(frame, clip_limit, tile_grid_size)

    result_no_clahe = model.predict(rgb_no_clahe, verbose=False)
    result_with_clahe = model.predict(rgb_with_clahe, verbose=False)

    return result_no_clahe[0].plot(), result_with_clahe[0].plot(), result_no_clahe[0], result_with_clahe[0]

# ==================== Hitung Deteksi ==================== #
def count_detections(result, original_frame):
    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()

    helmet_count = 0
    no_helmet_count = 0
    helm_crops = []
    non_helm_crops = []

    for box, cls in zip(xyxy, class_ids):
        x1, y1, x2, y2 = map(int, box)
        crop = original_frame[y1:y2, x1:x2]

        # Pastikan crop valid
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(crop)

        if cls == 0:
            helmet_count += 1
            helm_crops.append(pil_crop)
        elif cls == 1:
            no_helmet_count += 1
            non_helm_crops.append(pil_crop)

    return helmet_count, no_helmet_count, helm_crops, non_helm_crops

# ==================== Main App ==================== #
def main():
    show_header()
    video_path, clip_limit, tile_grid_size = sidebar_controls()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📽️ Tanpa CLAHE")
        frame_placeholder1 = st.empty()
        status_placeholder1 = st.empty()

        col1a, col1b = st.columns(2)
        with col1a:
            st.subheader("Helm")
            capture_placeholder1 = st.empty()
            
        with col1b:
            st.subheader("Tanpa Helm")
            capture_placeholder2 = st.empty()
        
    with col2:
        st.subheader("📽️✨ Dengan CLAHE")
        frame_placeholder2 = st.empty()
        status_placeholder2 = st.empty()
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.subheader("Helm")
            capture_placeholder3 = st.empty()
            
        with col2b:
            st.subheader("Tanpa Helm")
            capture_placeholder4 = st.empty()

    model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Tidak dapat membuka video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca frame dari video.")
            break

        if "record" in video_path.lower():
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_no_clahe, img_with_clahe,result_no_clahe, result_with_clahe  = process_frame(frame, model, clip_limit, tile_grid_size)
        helmet_count1, no_helmet_count1, helm_crops1, non_helm_crops1 = count_detections(result_no_clahe,frame)
        helmet_count2, no_helmet_count2, helm_crops2, non_helm_crops2 = count_detections(result_with_clahe,frame)

        # st.session_state.helmet_count = helmet_count

        # Tampilkan hasil deteksi
        frame_placeholder1.image(Image.fromarray(img_no_clahe))
        frame_placeholder2.image(Image.fromarray(img_with_clahe))

        #Tampilkan hasil capture
        capture_placeholder1.image(helm_crops1, width=50)
        capture_placeholder2.image(non_helm_crops1, width=50)
        capture_placeholder3.image(helm_crops2, width=50)
        capture_placeholder4.image(non_helm_crops2, width=50)

        # Update teks status secara dinamis tanpa menumpuk
        status_template = "Jumlah helm terdeteksi: **{}**, tanpa helm: **{}**"
        status_placeholder1.markdown(status_template.format(helmet_count1, no_helmet_count1))
        status_placeholder2.markdown(status_template.format(helmet_count2, no_helmet_count2))
        

    cap.release()

if __name__ == "__main__":
    main()
