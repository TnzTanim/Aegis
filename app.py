# app.py
import streamlit as st
import cv2
import numpy as np
import threading
import time
from PIL import Image

import public_security
import home_security
import aegis_utils as utils

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(page_title="Aegis", page_icon="🛡️", layout="wide")

# ------------------------------
# CSS for dark modern theme
# ------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0b1b33, #071027);
    color: #eaf2ff;
}
h1, h2, h3, h4 { color: #ffb86b; }
.card {
    background: #0b2a3a;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(2,6,23,0.7);
    text-align:center;
    transition: all 0.3s ease;
}
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 25px rgba(2,6,23,0.8);
}
.start-btn {
    background-color: #ffb86b;
    color: #071027;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 30px;
    margin-top: 10px;
    transition: all 0.3s ease;
}
.start-btn:hover {
    background-color: #ffa64d;
    cursor: pointer;
}
.stop-btn {
    background-color: #ff4c4c;
    color: #fff;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 30px;
    margin-top: 10px;
}
.stop-btn:hover {
    background-color: #ff2c2c;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1 style='text-align:center'>Aegis</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#98a0b3;'>Smart Public & Home Security Demo</h4>", unsafe_allow_html=True)

# ------------------------------
# Mode selection
# ------------------------------
st.markdown("<h3 style='text-align:center;margin-top:30px;'>Select Mode</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

if "mode_selected" not in st.session_state:
    st.session_state.mode_selected = None
if "running" not in st.session_state:
    st.session_state.running = False
if "thread_obj" not in st.session_state:
    st.session_state.thread_obj = None

with col1:
    if st.button("Public Surveillance", key="public_card"):
        st.session_state.mode_selected = "Public"
with col2:
    if st.button("Home Guardian", key="home_card"):
        st.session_state.mode_selected = "Home"

# ------------------------------
# Video and logs placeholders
# ------------------------------
video_display = st.empty()
alert_display = st.empty()

# show placeholder if not running
if not st.session_state.running:
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    video_display.image(placeholder_img, channels="BGR", caption="Video will appear here", use_column_width=True)

# ------------------------------
# Functions to run modes
# ------------------------------
CAMERA_INDEX = 0  # webcam

def run_public():
    try:
        ps = public_security.PublicSecurity(
            fire_model_path=r"C:\Users\TNZ\PycharmProjects\Aegis\fire.pt",
            violence_model_path=r"C:\Users\TNZ\PycharmProjects\Aegis\Violence.pt"
        )
    except Exception as e:
        alert_display.error(f"Failed to load models: {e}")
        st.session_state.running = False
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        alert_display.error("Cannot open camera.")
        st.session_state.running = False
        return

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            alert_display.error("Camera frame not available.")
            break
        annotated, detections = ps.infer_frame(frame)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_display.image(img, use_column_width=True)
        if detections:
            alert_display.info(", ".join([f"{d['label']} ({d['conf']:.2f})" for d in detections]))
        else:
            alert_display.empty()
        time.sleep(0.03)
    cap.release()

def run_home():
    try:
        hs = home_security.HomeSecurity(model_path="yolov8n.pt")
    except Exception as e:
        alert_display.error(f"Failed to load home model: {e}")
        st.session_state.running = False
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        alert_display.error("Cannot open camera.")
        st.session_state.running = False
        return

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            alert_display.error("Camera frame not available.")
            break
        annotated, detections, info = hs.process_frame(frame)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_display.image(img, use_column_width=True)
        if detections:
            alert_display.info(", ".join([f"{d['label']} ({d['conf']:.2f})" for d in detections]))
        else:
            alert_display.empty()
        if info.get("triggered"):
            alert_display.warning(f"⚠️ ALERT: Threat detected! Saved video and screenshot in /alerts/")
        time.sleep(0.03)
    cap.release()

# ------------------------------
# Start / Stop buttons
# ------------------------------
col_start, col_stop = st.columns([1,1])
with col_start:
    if st.button("Start", key="start_btn", help="Start security system", use_container_width=True):
        if st.session_state.mode_selected is None:
            alert_display.error("Please select a mode first!")
        else:
            st.session_state.running = True
            if st.session_state.mode_selected == "Public":
                t = threading.Thread(target=run_public, daemon=True)
            else:
                t = threading.Thread(target=run_home, daemon=True)
            st.session_state.thread_obj = t
            t.start()
with col_stop:
    if st.button("Stop", key="stop_btn", help="Stop security system", use_container_width=True):
        st.session_state.running = False
        alert_display.info("System stopped.")
