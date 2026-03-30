import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Config ---
CSV_FILE = "./training_data/labels.csv"
IMAGE_FOLDER = "./training_data/images"
POSE_MODEL_PATH = "yolov8n-pose.pt"
SAVE_PATH = "height_weight_model.pt"
OUTPUT_DIR = "./training_output"
KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, "keypoints")
EPOCHS = 200
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# --- Create output directories ---
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

# --- Load YOLOv8-Pose model ---
yolo_pose = YOLO(POSE_MODEL_PATH)


# --- Extract keypoints using YOLOv8-Pose ---
# COCO 17 keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def draw_keypoints_on_image(image_path, keypoints_xy, save_path):
    """Draw keypoints and skeleton on image and save."""
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # keypoints_xy is shape (17, 2) normalized
    pts = []
    for i in range(17):
        x = int(keypoints_xy[i][0] * w)
        y = int(keypoints_xy[i][1] * h)
        pts.append((x, y))
        if x > 0 or y > 0:  # only draw if detected
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{i}", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Draw skeleton
    for (a, b) in SKELETON:
        if (pts[a][0] > 0 or pts[a][1] > 0) and (pts[b][0] > 0 or pts[b][1] > 0):
            cv2.line(img, pts[a], pts[b], (0, 200, 255), 2)

    cv2.imwrite(save_path, img)


def extract_keypoints(image_path, save_vis=True):
    """Extract 17 keypoints (x, y) from YOLOv8-Pose → 34 features.
    Optionally save annotated image with keypoints drawn."""
    results = yolo_pose(image_path, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    # Take the first detected person's keypoints
    kps_raw = result.keypoints[0].xyn.cpu().numpy().squeeze()  # shape (17, 2)
    kps = kps_raw.flatten()  # normalized (x, y) * 17 = 34 values

    if len(kps) != 34 or np.all(kps == 0):
        return None

    # Save keypoint visualization
    if save_vis:
        img_name = os.path.basename(image_path)
        save_path = os.path.join(KEYPOINTS_DIR, f"keypoints_{img_name}")
        draw_keypoints_on_image(image_path, kps_raw, save_path)
        print(f"  Saved keypoints visualization → {save_path}")

    return kps


# --- PyTorch Regression Model ---
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
            nn.Linear(32, 2),  # output: [height, weight]
        )

    def forward(self, x):
        return self.net(x)


# --- Custom Dataset ---
class PoseDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Load Dataset ---
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} samples from {CSV_FILE}")

features_list = []
labels_list = []
skipped = 0

for _, row in df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row["image"])
    keypoints = extract_keypoints(img_path)

    if keypoints is None:
        print(f"  Skipped {row['image']} (no pose detected)")
        skipped += 1
        continue

    features_list.append(keypoints)
    labels_list.append([row["height"], row["weight"]])

print(f"Extracted features from {len(features_list)} images ({skipped} skipped)")

if len(features_list) < 2:
    print("Not enough samples to train. Add more labeled images to data/images/ and update data/labels.csv.")
    exit(1)

X = np.array(features_list)
y = np.array(labels_list)

# --- Normalize labels for better training ---
y_mean = y.mean(axis=0)
y_std = y.std(axis=0)
y_std[y_std == 0] = 1  # avoid division by zero
y_norm = (y - y_mean) / y_std

# --- Create DataLoader ---
dataset = PoseDataset(X, y_norm)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Train ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

model = HeightWeightNet(input_dim=34).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

model.train()
epoch_losses = []
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

# --- Save loss curve ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), epoch_losses, linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_plot_path, dpi=150)
plt.close()
print(f"\nLoss curve saved → {loss_plot_path}")

# --- Save epoch losses to CSV ---
loss_csv_path = os.path.join(OUTPUT_DIR, "epoch_losses.csv")
loss_df = pd.DataFrame({"epoch": range(1, EPOCHS + 1), "loss": epoch_losses})
loss_df.to_csv(loss_csv_path, index=False)
print(f"Epoch losses saved → {loss_csv_path}")

# --- Evaluate on full dataset ---
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X).to(device)
    preds_norm = model(X_tensor).cpu().numpy()
    preds = preds_norm * y_std + y_mean  # denormalize

print("\n--- Predictions vs Actual ---")
results_rows = []
for i in range(len(y)):
    print(f"  Actual: H={y[i][0]:.1f}cm W={y[i][1]:.1f}kg | "
          f"Predicted: H={preds[i][0]:.1f}cm W={preds[i][1]:.1f}kg")
    results_rows.append({
        "image": df.iloc[i]["image"] if i < len(df) else f"sample_{i}",
        "actual_height": y[i][0],
        "actual_weight": y[i][1],
        "predicted_height": round(float(preds[i][0]), 2),
        "predicted_weight": round(float(preds[i][1]), 2),
        "height_error": round(abs(float(preds[i][0]) - y[i][0]), 2),
        "weight_error": round(abs(float(preds[i][1]) - y[i][1]), 2),
    })

# Save predictions to CSV
results_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
print(f"\nPredictions saved → {results_csv_path}")

# --- Save predictions comparison plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(len(y)), y[:, 0], alpha=0.6, label="Actual", width=0.4)
axes[0].bar([x + 0.4 for x in range(len(y))], preds[:, 0], alpha=0.6, label="Predicted", width=0.4)
axes[0].set_title("Height (cm)")
axes[0].set_xlabel("Sample")
axes[0].legend()

axes[1].bar(range(len(y)), y[:, 1], alpha=0.6, label="Actual", width=0.4)
axes[1].bar([x + 0.4 for x in range(len(y))], preds[:, 1], alpha=0.6, label="Predicted", width=0.4)
axes[1].set_title("Weight (kg)")
axes[1].set_xlabel("Sample")
axes[1].legend()

plt.tight_layout()
pred_plot_path = os.path.join(OUTPUT_DIR, "predictions_comparison.png")
plt.savefig(pred_plot_path, dpi=150)
plt.close()
print(f"Predictions plot saved → {pred_plot_path}")

# --- Save model + normalization stats ---
torch.save({
    "model_state": model.cpu().state_dict(),
    "y_mean": torch.tensor(y_mean, dtype=torch.float32),
    "y_std": torch.tensor(y_std, dtype=torch.float32),
}, SAVE_PATH)
print(f"\nModel saved to {SAVE_PATH}")
