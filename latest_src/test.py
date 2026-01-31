import torch, cv2, os, numpy as np
from torch.utils.data import DataLoader
from latest_src.dataset import SegmentationDataset
from model import get_model
from transform import get_val_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6

model = get_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("outputs/models/best_model.pth"))
model.eval()

test_ds = SegmentationDataset(
    r"data\Offroad_Segmentation_testImages\Color_Images",
    r"data\Offroad_Segmentation_testImages\Color_Images",
    transform=get_val_transforms()
)

loader = DataLoader(test_ds, batch_size=1)

os.makedirs("outputs/preds", exist_ok=True)

with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        img = img.to(DEVICE)
        pred = model(img).argmax(1).cpu().numpy()[0]
        cv2.imwrite(f"outputs/preds/pred_{i:04d}.png", pred.astype(np.uint8))

print("✅ Test predictions saved")
