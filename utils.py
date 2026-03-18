import numpy as np

def point_xy(landmarks, idx, w, h):
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def distance(a, b):
    return np.linalg.norm(a - b)