
import cv2
import time
from collections import deque
from ultralytics import YOLO
from aegis_utils import save_frames_as_video, safe_timestamp_name, log_info, log_warn, ensure_dir

class HomeSecurity:
    """
    Lightweight home guardian using yolov8n (auto-download if needed).
    - buffer_seconds: how many seconds of past frames to keep
    - fps: frames per second assumed
    - trigger_frames: consecutive frames with detection to trigger
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.35, buffer_seconds=5, fps=15, trigger_frames=3, imgsz=320):
        self.conf = conf
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.trigger_frames = trigger_frames
        self.imgsz = imgsz

        log_info(f"Loading home model ({model_path}) — will auto-download if missing.")
        self.model = YOLO(model_path)  # yolov8n.pt will be downloaded automatically by ultralytics if absent

        # class names (COCO) are inside the model
        try:
            self.names = self.model.model.names
        except Exception:
            self.names = {}

        self.frame_buffer = deque(maxlen=int(self.buffer_seconds * self.fps))
        self.detection_counter = 0

    def _parse_results(self, results):
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for i in range(len(boxes.xyxy)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
                detections.append({"label": label, "conf": conf, "box": xyxy})
        return detections

    def process_frame(self, frame):
        """
        Returns annotated frame, detections list, info dict:
          info = {'triggered': bool, 'video_path': str or None, 'screenshot': str or None}
        """
        # Add to circular buffer
        self.frame_buffer.append(frame.copy())

        # Run model
        results = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz)
        detections = self._parse_results(results)

        # annotate
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            txt = f"{d['label']} {d['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, txt, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # trigger logic
        if detections:
            self.detection_counter += 1
        else:
            self.detection_counter = 0

        if self.detection_counter >= self.trigger_frames:
            # prepare paths
            ts_name = safe_timestamp_name("home_alert")
            ensure_dir("alerts")
            video_path = f"alerts/{ts_name}.mp4"
            screenshot_path = f"alerts/{ts_name}.jpg"
            frames = list(self.frame_buffer)
            saved = save_frames_as_video(frames, video_path, fps=self.fps)
            # last frame as screenshot
            if frames:
                cv2.imwrite(screenshot_path, frames[-1])
            log_warn(f"Home alert triggered. Saved: {video_path} (success={saved}), screenshot: {screenshot_path}")
            self.detection_counter = 0
            return annotated, detections, {"triggered": True, "video_path": video_path, "screenshot": screenshot_path}

        return annotated, detections, {"triggered": False}
