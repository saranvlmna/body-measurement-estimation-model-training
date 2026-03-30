import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
image_path='./image.jpeg'

def extract_keypoints(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    keypoints = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y))

    return keypoints

import numpy as np

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_features(keypoints):
    head = keypoints[0]
    ankle = keypoints[28]
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    hip = keypoints[24]

    height_px = distance(head, ankle)
    shoulder_width = distance(left_shoulder, right_shoulder)
    torso_ratio = distance(head, hip) / height_px

    return [height_px, shoulder_width, torso_ratio]

keypoints = extract_keypoints(image_path)
features = extract_features(keypoints)
# print("Extracted Features:", features)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample training data — replace with real measurements
data = {
    "height_px": [0.65, 0.70, 0.60, 0.68, 0.55, 0.72, 0.63, 0.67],
    "shoulder_width": [0.22, 0.25, 0.20, 0.24, 0.19, 0.26, 0.21, 0.23],
    "torso_ratio": [0.53, 0.50, 0.55, 0.51, 0.56, 0.49, 0.54, 0.52],
    "height": [170, 180, 160, 175, 155, 185, 165, 178],
    "weight": [65, 80, 55, 75, 50, 85, 60, 72],
}
df = pd.DataFrame(data)

X = df[["height_px", "shoulder_width", "torso_ratio"]]
y = df[["height", "weight"]]

model = RandomForestRegressor()
model.fit(X, y)

pred = model.predict([features])
print("Predicted Height & Weight:", pred)