# aegis_utils.py
import os
import cv2
import time
import json
from collections import deque

def save_frames_as_video(frames, out_path, fps=15):
    """Save list of BGR frames as mp4 file. Returns True if success."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return os.path.exists(out_path)

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def safe_timestamp_name(prefix="aegis"):
    ts = int(time.time())
    return f"{prefix}_{ts}"

def log_info(msg):
    print(f"[INFO] {time.ctime()} - {msg}")

def log_warn(msg):
    print(f"[WARN] {time.ctime()} - {msg}")

def log_error(msg):
    print(f"[ERROR] {time.ctime()} - {msg}")
