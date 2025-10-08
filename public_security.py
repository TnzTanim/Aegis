# public_security.py
import cv2
import numpy as np
from ultralytics import YOLO
from aegis_utils import log_info

class PublicSecurity:
    """
    Loads two YOLOv8 models and runs them per-frame.
    - fire_model_path: path to your fire.pt
    - violence_model_path: path to your Violence.pt
    """

    def __init__(self, fire_model_path, violence_model_path, conf=0.25, imgsz=640):
        self.conf = conf
        self.imgsz = imgsz
        log_info(f"Loading fire model from: {fire_model_path}")
        self.fire_model = YOLO(fire_model_path)
        log_info(f"Loading violence model from: {violence_model_path}")
        self.violence_model = YOLO(violence_model_path)

        # get class name mappings (YOLOv8 stores them)
        try:
            self.fire_names = self.fire_model.model.names
        except Exception:
            self.fire_names = {0: "fire"}
        try:
            self.violence_names = self.violence_model.model.names
        except Exception:
            self.violence_names = {0: "NonViolence", 1: "Violence"}

    def _parse_results(self, results, names_map, tag):
        """
        results: list of Results from ultralytics predict
        returns list of detections: dicts with label, conf, box
        """
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            # boxes is a Boxes object; iterate per box
            for i in range(len(boxes.xyxy)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()  # [x1,y1,x2,y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = names_map.get(cls_id, str(cls_id)) if isinstance(names_map, dict) else names_map[cls_id]
                detections.append({
                    "label": f"{tag}:{label}",
                    "conf": conf,
                    "box": xyxy
                })
        return detections

    def infer_frame(self, frame):
        """
        frame: BGR numpy array
        returns annotated_frame (BGR), detections (list)
        """
        # run both models
        # using predict ensures we can pass conf and imgsz
        res_fire = self.fire_model.predict(source=frame, conf=self.conf, imgsz=self.imgsz)
        res_violence = self.violence_model.predict(source=frame, conf=self.conf, imgsz=self.imgsz)

        dets_fire = self._parse_results(res_fire, self.fire_names, "fire")
        dets_violence = self._parse_results(res_violence, self.violence_names, "violence")

        detections = dets_fire + dets_violence

        # annotate
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label_text = f"{d['label']} {d['conf']:.2f}"
            color = (0, 0, 255) if d["label"].startswith("fire") else (0, 165, 255)  # red for fire, orange for violence
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return annotated, detections
