import io
import tempfile
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

# --- YOLOv8-Pose model ---
POSE_MODEL_PATH = "yolov8n-pose.pt"
yolo_pose = YOLO(POSE_MODEL_PATH)


# --- PyTorch Regression Model (must match training architecture) ---
class HeightWeightNet(nn.Module):
    def __init__(self, input_dim=34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


# --- Load trained model ---
MODEL_PATH = "height_weight_model.pt"
model = None
y_mean = None
y_std = None

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = HeightWeightNet(input_dim=34)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_mean = checkpoint["y_mean"].numpy()
    y_std = checkpoint["y_std"].numpy()


def extract_keypoints_from_bytes(image_bytes: bytes):
    """Decode image bytes, save to temp file, run YOLOv8-Pose."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # YOLOv8 needs a file path or numpy array
    results = yolo_pose(img, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    kps = result.keypoints[0].xyn.cpu().numpy().flatten()

    if len(kps) != 34 or np.all(kps == 0):
        return None

    return kps


@app.post("/predict")
async def predict_height_weight(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first by running model.train.py")

    contents = await image.read()

    keypoints = extract_keypoints_from_bytes(contents)
    if keypoints is None:
        raise HTTPException(status_code=400, detail="No person/pose detected in the image")

    with torch.no_grad():
        x = torch.FloatTensor(keypoints).unsqueeze(0)
        pred_norm = model(x).numpy()[0]
        # Denormalize
        height = float(pred_norm[0] * y_std[0] + y_mean[0])
        weight = float(pred_norm[1] * y_std[1] + y_mean[1])

    return {
        "height": round(height, 2),
        "weight": round(weight, 2),
    }
