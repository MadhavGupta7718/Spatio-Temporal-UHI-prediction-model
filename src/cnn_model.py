"""
CNN Land Cover Classification Module
======================================
ResNet-50 Transfer Learning for satellite image classification.

Classes (5):
  0 — Dense Concrete (buildings, commercial)
  1 — Asphalt Roads (highways, streets)
  2 — Dense Vegetation (forests, parks)
  3 — Sparse Vegetation (crops, grassland)
  4 — Water Bodies (rivers, lakes)

Training approach:
  - Pre-trained ResNet-50 on ImageNet
  - Replace final FC layer: nn.Linear(2048, 5)
  - Freeze early layers → train last 2 blocks first
  - Unfreeze all after epoch 10 for fine-tuning
  - Use EuroSAT dataset (remapped to 5 classes)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CNN_CONFIG, EUROSAT_CLASS_MAP, MODELS_DIR, RANDOM_SEED
from src.utils import setup_logger, set_seed, ensure_dir

logger = setup_logger("cnn_model")


# ══════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════

class LandCoverDataset(Dataset):
    """
    PyTorch Dataset for land cover satellite image patches.
    
    Supports two modes:
      1. EuroSAT mode: loads from EuroSAT directory structure
         EuroSAT/
           ├── Industrial/
           ├── Residential/
           ├── Highway/
           └── ...
      2. Custom mode: loads from a CSV with columns [image_path, label]
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, class_names: List[str] = None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of integer class labels
            transform: torchvision transforms to apply
            class_names: List of class name strings
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or CNN_CONFIG["class_names"]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: int = None, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.
    
    Training: Random augmentations (flip, rotate, color jitter)
    Evaluation: Just resize + normalize
    """
    image_size = image_size or CNN_CONFIG["image_size"]
    
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def load_eurosat_dataset(eurosat_dir: Path) -> Tuple[List[str], List[int]]:
    """
    Load EuroSAT dataset and remap classes to our 5-class scheme.
    
    Args:
        eurosat_dir: Path to EuroSAT root directory
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    for eurosat_class, our_label in EUROSAT_CLASS_MAP.items():
        class_dir = eurosat_dir / eurosat_class
        if not class_dir.exists():
            logger.warning(f"EuroSAT class directory not found: {class_dir}")
            continue
        
        for img_file in class_dir.glob("*.jpg"):
            image_paths.append(str(img_file))
            labels.append(our_label)
        
        for img_file in class_dir.glob("*.png"):
            image_paths.append(str(img_file))
            labels.append(our_label)
    
    logger.info(f"Loaded {len(image_paths)} images from EuroSAT")
    
    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls} ({CNN_CONFIG['class_names'][cls]}): {count} images")
    
    return image_paths, labels


def generate_synthetic_dataset(n_samples: int = 5000, 
                                 image_size: int = None,
                                 seed: int = RANDOM_SEED) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic satellite-like image patches for demo/testing.
    
    Each class has distinct color patterns:
      - Dense Concrete: Gray/brown tones
      - Asphalt Roads: Dark gray, linear patterns
      - Dense Vegetation: Deep green
      - Sparse Vegetation: Light green/yellow
      - Water Bodies: Blue/dark blue
    """
    np.random.seed(seed)
    image_size = image_size or CNN_CONFIG["image_size"]
    
    images = []
    labels = []
    
    samples_per_class = n_samples // CNN_CONFIG["num_classes"]
    
    color_profiles = {
        0: {"mean": [160, 140, 130], "std": [20, 20, 20]},   # Dense Concrete
        1: {"mean": [80, 80, 85], "std": [15, 15, 15]},      # Asphalt
        2: {"mean": [30, 120, 40], "std": [15, 25, 15]},     # Dense Vegetation
        3: {"mean": [100, 160, 70], "std": [20, 20, 20]},    # Sparse Vegetation
        4: {"mean": [40, 60, 140], "std": [15, 15, 25]},     # Water
    }
    
    for class_idx, profile in color_profiles.items():
        for _ in range(samples_per_class):
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            
            for c in range(3):
                channel = np.random.normal(
                    profile["mean"][c], profile["std"][c],
                    (image_size, image_size)
                )
                # Add spatial texture
                channel += np.random.normal(0, 5, (image_size, image_size))
                img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            
            images.append(img)
            labels.append(class_idx)
    
    logger.info(f"Generated {len(images)} synthetic satellite patches")
    return images, labels


# ══════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ══════════════════════════════════════════════

class UHILandCoverCNN(nn.Module):
    """
    ResNet-50 based land cover classifier.
    
    Architecture:
      - Pre-trained ResNet-50 backbone
      - Custom classification head:
          AdaptiveAvgPool → Flatten → Dropout → FC(2048, 512) → ReLU 
          → Dropout → FC(512, num_classes)
    """
    
    def __init__(self, num_classes: int = None, pretrained: bool = True):
        super(UHILandCoverCNN, self).__init__()
        
        num_classes = num_classes or CNN_CONFIG["num_classes"]
        
        # Load pre-trained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 2048 for ResNet-50
        
        # Replace the classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Built ResNet-50 model with {num_classes} output classes")
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen — only FC head will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — all layers will be fine-tuned")
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════
# 3. TRAINING ENGINE
# ══════════════════════════════════════════════

class CNNTrainer:
    """
    Manages the training loop, validation, and model checkpointing.
    """
    
    def __init__(self, model: UHILandCoverCNN, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        logger.info(f"Training device: {self.device}")
        logger.info(f"Trainable parameters: {self.model.get_trainable_params():,}")
    
    def setup_optimizer(self, lr: float = None):
        """Configure optimizer and learning rate scheduler."""
        lr = lr or CNN_CONFIG["learning_rate"]
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=CNN_CONFIG["weight_decay"]
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = None, unfreeze_after: int = None) -> Dict:
        """
        Full training loop with:
          - Progressive unfreezing
          - Learning rate scheduling
          - Best model checkpointing
          - Training history tracking
        """
        epochs = epochs or CNN_CONFIG["epochs"]
        unfreeze_after = unfreeze_after or CNN_CONFIG["unfreeze_after_epoch"]
        
        self.setup_optimizer()
        
        # Start with frozen backbone
        if CNN_CONFIG["freeze_layers"]:
            self.model.freeze_backbone()
            self.setup_optimizer(lr=CNN_CONFIG["learning_rate"])
        
        best_val_acc = 0.0
        ensure_dir(MODELS_DIR)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Will unfreeze backbone after epoch {unfreeze_after}")
        
        for epoch in range(1, epochs + 1):
            # Progressive unfreezing
            if epoch == unfreeze_after + 1:
                self.model.unfreeze_backbone()
                self.setup_optimizer(lr=CNN_CONFIG["learning_rate"] * 0.1)
                logger.info(f"Epoch {epoch}: Unfroze backbone, reduced LR")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Schedule LR
            self.scheduler.step(val_loss)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": CNN_CONFIG["class_names"],
                }, MODELS_DIR / "best_land_cover_cnn.pth")
                logger.info(f"  → Saved best model (val_acc: {val_acc:.4f})")
        
        logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
        return self.history
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[int, np.ndarray]:
        """
        Predict land cover class for a single image.
        
        Returns:
            Tuple of (predicted_class_index, class_probabilities)
        """
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)
        
        outputs = self.model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def load_checkpoint(self, checkpoint_path: Path = None):
        """Load a saved model checkpoint."""
        checkpoint_path = checkpoint_path or MODELS_DIR / "best_land_cover_cnn.pth"
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"  Val accuracy: {checkpoint['val_acc']:.4f}")


# ══════════════════════════════════════════════
# 4. INFERENCE HELPERS
# ══════════════════════════════════════════════

def classify_grid_patches(model: UHILandCoverCNN, 
                           patches: List[np.ndarray],
                           device: str = "cpu") -> pd.DataFrame:
    """
    Classify a batch of image patches and return results.
    
    Args:
        model: Trained CNN model
        patches: List of image arrays (H, W, 3)
        device: Computing device
    
    Returns:
        DataFrame with columns [predicted_class, class_name, confidence, probabilities]
    """
    model.eval()
    model.to(device)
    
    transform = get_transforms(is_training=False)
    results = []
    
    for patch in tqdm(patches, desc="Classifying patches"):
        img = Image.fromarray(patch.astype(np.uint8))
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        pred_class = np.argmax(probs)
        results.append({
            "predicted_class": pred_class,
            "class_name": CNN_CONFIG["class_names"][pred_class],
            "confidence": probs[pred_class],
            "probabilities": probs.tolist(),
        })
    
    return pd.DataFrame(results)


# ══════════════════════════════════════════════
# 5. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_cnn_training(eurosat_dir: Optional[Path] = None,
                      use_synthetic: bool = False) -> Dict:
    """
    Run the full CNN training pipeline.
    
    Args:
        eurosat_dir: Path to EuroSAT dataset directory
        use_synthetic: If True, use synthetic data for demo
    
    Returns:
        Training history dictionary
    """
    set_seed(RANDOM_SEED)
    
    logger.info("=" * 60)
    logger.info("STARTING CNN TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load dataset
    if use_synthetic or eurosat_dir is None:
        logger.info("Using synthetic dataset for demo")
        images, labels = generate_synthetic_dataset(n_samples=2000)
        
        # Create temporary dataset from numpy arrays
        transform = get_transforms(is_training=True)
        
        class NumpyDataset(Dataset):
            def __init__(self, images, labels, transform):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img = Image.fromarray(self.images[idx])
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]
        
        dataset = NumpyDataset(images, labels, transform)
    else:
        image_paths, labels = load_eurosat_dataset(eurosat_dir)
        transform = get_transforms(is_training=True)
        dataset = LandCoverDataset(image_paths, labels, transform)
    
    # Step 2: Split dataset
    n_total = len(dataset)
    n_train = int(n_total * CNN_CONFIG["train_split"])
    n_val = int(n_total * CNN_CONFIG["val_split"])
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_set, batch_size=CNN_CONFIG["batch_size"], 
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=CNN_CONFIG["batch_size"], 
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=CNN_CONFIG["batch_size"], 
                              shuffle=False, num_workers=0)
    
    logger.info(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Step 3: Build model
    model = UHILandCoverCNN(pretrained=CNN_CONFIG["pretrained"])
    
    # Step 4: Train
    trainer = CNNTrainer(model)
    history = trainer.train(train_loader, val_loader)
    
    # Step 5: Evaluate on test set
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"\nTest Set Results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
    
    return history


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Land Cover CNN")
    parser.add_argument("--eurosat-dir", type=str, default=None,
                        help="Path to EuroSAT dataset directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for demo")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.epochs:
        CNN_CONFIG["epochs"] = args.epochs
    
    history = run_cnn_training(
        eurosat_dir=Path(args.eurosat_dir) if args.eurosat_dir else None,
        use_synthetic=args.synthetic
    )
