import math
import time
import json
import os
import numpy as np
from collections import deque
from utils import point_xy, distance

# MediaPipe FaceMesh landmark index
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

LEFT_FACE = 234
RIGHT_FACE = 454

MOTION_POINTS = [33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14, 1, 152]


class ParkinsonRiskAnalyzer:
    def __init__(self):
        self.ear_history = deque(maxlen=300)
        self.mouth_history = deque(maxlen=300)
        self.tilt_history = deque(maxlen=300)
        self.motion_history = deque(maxlen=300)

        self.blink_count = 0
        self.blink_closed = False
        self.start_time = time.time()
        self.prev_motion_pts = None

    def eye_aspect_ratio(self, landmarks, w, h, side="left"):
        if side == "left":
            top = point_xy(landmarks, LEFT_EYE_TOP, w, h)
            bottom = point_xy(landmarks, LEFT_EYE_BOTTOM, w, h)
            left = point_xy(landmarks, LEFT_EYE_LEFT, w, h)
            right = point_xy(landmarks, LEFT_EYE_RIGHT, w, h)
        else:
            top = point_xy(landmarks, RIGHT_EYE_TOP, w, h)
            bottom = point_xy(landmarks, RIGHT_EYE_BOTTOM, w, h)
            left = point_xy(landmarks, RIGHT_EYE_LEFT, w, h)
            right = point_xy(landmarks, RIGHT_EYE_RIGHT, w, h)

        vertical = distance(top, bottom)
        horizontal = distance(left, right) + 1e-6
        return float(vertical / horizontal)

    def mouth_ratio(self, landmarks, w, h):
        left = point_xy(landmarks, MOUTH_LEFT, w, h)
        right = point_xy(landmarks, MOUTH_RIGHT, w, h)
        top = point_xy(landmarks, MOUTH_TOP, w, h)
        bottom = point_xy(landmarks, MOUTH_BOTTOM, w, h)

        vertical = distance(top, bottom)
        horizontal = distance(left, right) + 1e-6
        return float(vertical / horizontal)

    def head_tilt_angle(self, landmarks, w, h):
        left_face = point_xy(landmarks, LEFT_FACE, w, h)
        right_face = point_xy(landmarks, RIGHT_FACE, w, h)
        dx = right_face[0] - left_face[0]
        dy = right_face[1] - left_face[1]
        return float(math.degrees(math.atan2(dy, dx)))

    def facial_motion_energy(self, curr_pts):
        if self.prev_motion_pts is None:
            self.prev_motion_pts = curr_pts
            return 0.0

        motion = float(np.mean(np.linalg.norm(curr_pts - self.prev_motion_pts, axis=1)))
        self.prev_motion_pts = curr_pts
        return motion

    def update(self, landmarks, w, h):
        left_ear = self.eye_aspect_ratio(landmarks, w, h, "left")
        right_ear = self.eye_aspect_ratio(landmarks, w, h, "right")
        ear = (left_ear + right_ear) / 2.0

        mouth = self.mouth_ratio(landmarks, w, h)
        tilt = self.head_tilt_angle(landmarks, w, h)

        curr_motion_pts = np.array(
            [point_xy(landmarks, idx, w, h) for idx in MOTION_POINTS],
            dtype=np.float32
        )
        motion = self.facial_motion_energy(curr_motion_pts)

        self.ear_history.append(ear)
        self.mouth_history.append(mouth)
        self.tilt_history.append(tilt)
        self.motion_history.append(motion)

        if ear < 0.18 and not self.blink_closed:
            self.blink_closed = True
        elif ear >= 0.18 and self.blink_closed:
            self.blink_closed = False
            self.blink_count += 1

        elapsed_min = max((time.time() - self.start_time) / 60.0, 1e-6)
        blink_per_min = self.blink_count / elapsed_min

        mean_ear = float(np.mean(self.ear_history)) if self.ear_history else 0.0
        mouth_std = float(np.std(self.mouth_history)) if len(self.mouth_history) > 10 else 0.0
        tilt_std = float(np.std(self.tilt_history)) if len(self.tilt_history) > 10 else 0.0
        motion_mean = float(np.mean(self.motion_history)) if self.motion_history else 0.0

        # heuristic risk score
        blink_score = max(0.0, min((15.0 - blink_per_min) / 15.0, 1.0))
        mouth_score = max(0.0, min((0.015 - mouth_std) / 0.015, 1.0))
        motion_score = max(0.0, min((1.8 - motion_mean) / 1.8, 1.0))
        tilt_score = max(0.0, min((2.0 - tilt_std) / 2.0, 1.0))

        risk_score = (
            0.35 * blink_score
            + 0.25 * mouth_score
            + 0.25 * motion_score
            + 0.15 * tilt_score
        )

        if risk_score > 0.70:
            label = "HIGH RISK"
        elif risk_score > 0.45:
            label = "MODERATE"
        else:
            label = "LOW"

        return {
            "blink_per_min": round(blink_per_min, 2),
            "mean_ear": round(mean_ear, 4),
            "mouth_std": round(mouth_std, 4),
            "tilt_std": round(tilt_std, 4),
            "motion_mean": round(motion_mean, 4),
            "risk_score": round(risk_score, 4),
            "risk_label": label
        }

    def save_result(self, result, path="results/session_logs/last_result.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)