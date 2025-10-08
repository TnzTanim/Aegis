# demo_aegis_advanced.py
import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Aegis", page_icon="🛡️", layout="wide")

# ---------------------------
# CSS for advanced dark theme
# ---------------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #0b1b33, #071027); color: #eaf2ff; }
h1,h2,h3 { color: #ffb86b; text-align:center; }
.big-btn {
    background:#ffb86b; color:#071027; font-size:22px; font-weight:700;
    padding:20px 0; border-radius:14px; width:220px; margin:12px; border:none;
    transition: all 0.3s ease;
}
.big-btn:hover { background:#ffa64d; transform:scale(1.05); cursor:pointer;}
.mode-card {
    background:#0b2a3a; border-radius:14px; padding:24px; text-align:center;
    transition: all 0.3s ease; box-shadow: 0 8px 20px rgba(2,6,23,0.7);
}
.mode-card:hover { transform:scale(1.03); box-shadow:0 12px 28px rgba(2,6,23,0.8); cursor:pointer;}
.stop-btn { background:#ff4c4c; color:white; font-size:18px; font-weight:700; padding:12px 20px; border-radius:12px; border:none; margin-top:12px;}
.stop-btn:hover { background:#ff2c2c; cursor:pointer; transform:scale(1.03);}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session state
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"  # landing -> mode -> boot -> live
if "mode" not in st.session_state:
    st.session_state.mode = None
if "running" not in st.session_state:
    st.session_state.running = False
if "video_placeholder" not in st.session_state:
    st.session_state.video_placeholder = None
if "last_checked" not in st.session_state:
    st.session_state.last_checked = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Helper functions
# ---------------------------
def render_header():
    st.markdown("<h1>Aegis</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Smart Public & Home Security System</h3>", unsafe_allow_html=True)

def black_placeholder(h=360, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

# ---------------------------
# Landing page
# ---------------------------
def landing_page():
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; font-size:16px; color:#98a0b3;'>Aegis provides fast, reliable monitoring for public and home environments. Click Start to continue.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    start = st.button("Start", key="start_demo")
    if start:
        st.session_state.page = "mode"

# ---------------------------
# Mode selection page
# ---------------------------
def mode_page():
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#98a0b3; margin-bottom:16px;'>Select a mode to begin monitoring.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Public Surveillance", key="public_mode"):
            st.session_state.mode = "Public Surveillance"
            st.session_state.page = "boot"
    with col2:
        if st.button("Home Guardian", key="home_mode"):
            st.session_state.mode = "Home Guardian"
            st.session_state.page = "boot"
    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Boot / initialization animation
# ---------------------------
def boot_page():
    render_header()
    st.markdown("<h3 style='text-align:center; color:#ffb86b;'>Initializing Aegis...</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#98a0b3;'>Preparing systems. Please wait.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    boot_steps = [
        "Checking camera access",
        "Loading modules",
        "Optimizing performance",
        "Finalizing setup"
    ]
    for i, step in enumerate(boot_steps):
        status_placeholder.info(step + "...")
        progress_placeholder.progress(int((i+1)/len(boot_steps)*100))
        time.sleep(0.8)
    progress_placeholder.empty()
    status_placeholder.success("System ready.")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Proceed to Live Feed"):
        st.session_state.page = "live"
        st.session_state.last_checked = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.running = True

# ---------------------------
# Live webcam page
# ---------------------------
def live_page():
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        if st.session_state.video_placeholder is None:
            st.session_state.video_placeholder = st.empty()
        video_frame = st.session_state.video_placeholder
    with col2:
        st.markdown("<div style='background:#081b28;padding:16px;border-radius:12px;color:#eaf2ff;'>", unsafe_allow_html=True)
        st.markdown(f"<h4>Mode: {st.session_state.mode}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>Status: Running</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>Monitoring environment...</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>Last checked: {st.session_state.last_checked}</h4>", unsafe_allow_html=True)
        if st.button("Stop Monitoring", key="stop_btn"):
            st.session_state.running = False
            st.session_state.page = "mode"
            st.session_state.mode = None
        st.markdown("</div>", unsafe_allow_html=True)

    # Webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        video_frame.error("Cannot access webcam. Ensure no other app is using it.")
        st.session_state.running = False
        return

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                video_frame.error("Cannot read webcam feed.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame.image(frame_rgb, use_column_width=True)
            st.session_state.last_checked = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time.sleep(0.03)
    finally:
        cap.release()
        video_frame.image(black_placeholder(), caption="Stream stopped", use_column_width=True)
        st.session_state.running = False

# ---------------------------
# Page router
# ---------------------------
if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "mode":
    mode_page()
elif st.session_state.page == "boot":
    boot_page()
elif st.session_state.page == "live":
    live_page()
else:
    st.session_state.page = "landing"
    landing_page()
