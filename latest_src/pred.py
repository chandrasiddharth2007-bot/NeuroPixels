import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from latest_src.dataset import SegmentationDataset
from model import get_model
from transform import get_val_transforms

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
MODEL_PATH = "outputs/models/best_model.pth"

IMG_DIR = r"data\Offroad_Segmentation_Training_Dataset\val\Color_Images"
MASK_DIR = r"data\Offroad_Segmentation_Training_Dataset\val\Segmentation"
# ---------------------------------------

# Load dataset
dataset = SegmentationDataset(
    image_dir=IMG_DIR,
    mask_dir=MASK_DIR,
    transform=get_val_transforms()
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model
model = get_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Get one sample
image, gt_mask = next(iter(loader))
image = image.to(DEVICE)
gt_mask = gt_mask.squeeze().cpu().numpy()

# Predict
with torch.no_grad():
    output = model(image)
    pred_mask = output.argmax(1).squeeze().cpu().numpy()

# ---------------- Pixel Accuracy ----------------
correct = (pred_mask == gt_mask).sum()
total = gt_mask.size
pixel_accuracy = correct / total

print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

# ---------------- Visualization ----------------
image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()

plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.title("Input Image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth Mask")
plt.imshow(gt_mask, cmap="tab10")
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Prediction Mask\nPixel Acc: {pixel_accuracy:.3f}")
plt.imshow(pred_mask, cmap="tab10")
plt.axis("off")

plt.tight_layout()
plt.show()
