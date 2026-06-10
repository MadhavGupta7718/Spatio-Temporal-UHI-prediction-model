"""Check the saved CNN model checkpoint."""
import torch

ckpt = torch.load(
    r"d:\Spatio-Temporal UHI prediction model\data\models\best_land_cover_cnn.pth",
    map_location="cpu",
    weights_only=False
)

print(f"Epoch saved: {ckpt['epoch']}")
print(f"Val Accuracy: {ckpt['val_acc']:.4f}")
print(f"Val Loss: {ckpt['val_loss']:.4f}")
print(f"Classes: {ckpt['class_names']}")
print(f"Keys in checkpoint: {list(ckpt.keys())}")
