import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from latest_src.dataset import SegmentationDataset
from transform import get_train_transforms, get_val_transforms
from model import get_model
from training import train_one_epoch
from eval import evaluate
from utils import save_model

NUM_CLASSES = 6
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_ds = SegmentationDataset(
    r"data\Offroad_Segmentation_Training_Dataset\train\Color_Images",
    r"data\Offroad_Segmentation_Training_Dataset\train\Segmentation",
    transform=get_train_transforms()
)

val_ds = SegmentationDataset(
    r"data\Offroad_Segmentation_Training_Dataset\val\Color_Images",
    r"data\Offroad_Segmentation_Training_Dataset\val\Segmentation",
    transform=get_val_transforms()
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = get_model(NUM_CLASSES).to(device)

# 🔒 FREEZE ENCODER
for p in model.encoder.parameters():
    p.requires_grad = False
print("✅ Encoder frozen")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=0)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)

def loss_fn(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)

scaler = torch.cuda.amp.GradScaler()

best_iou = 0.0

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    iou = evaluate(model, val_loader, device, NUM_CLASSES)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | mIoU: {iou:.4f}")

    if iou > best_iou:
        best_iou = iou
        save_model(model, "outputs/models/best_model.pth")

print("🔥 Training complete. Best mIoU:", best_iou)
