"""
AgriSense - Model Training Script
==================================
Fine-tunes EfficientNetB3 on PlantVillage dataset (38 classes)
Based on: manthan89-py/Plant-Disease-Detection (enhanced with transfer learning)

Author: SEAI Individual Project
SDG Alignment: SDG 2 - Zero Hunger
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir": "./dataset/PlantVillage",
    "model_save_path": "./model/efficientnet_b3.pth",
    "class_names_path": "./model/class_names.json",
    "num_classes": 38,
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 15,
    "learning_rate": 0.001,
    "train_split": 0.8,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"[AgriSense] Using device: {CONFIG['device']}")
print(f"[AgriSense] PyTorch version: {torch.__version__}")


# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    """
    Data augmentation transforms for training and validation.
    Training: augmented for generalization
    Validation: only normalize (no augmentation)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"] + 32, CONFIG["img_size"] + 32)),
        transforms.RandomCrop(CONFIG["img_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # ImageNet normalization (EfficientNet was pretrained on ImageNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


# ─────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────
def load_dataset():
    """
    Loads PlantVillage dataset from disk.
    Expects folder structure:
        dataset/PlantVillage/
            Apple___Apple_scab/
            Apple___Black_rot/
            ...
    """
    train_tf, val_tf = get_transforms()

    # Load full dataset with training transforms first
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=train_tf)

    # Save class names for inference
    class_names = full_dataset.classes
    os.makedirs("./model", exist_ok=True)
    with open(CONFIG["class_names_path"], "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[AgriSense] Classes saved: {len(class_names)} classes")

    # Train/Validation split
    total_size = len(full_dataset)
    train_size = int(CONFIG["train_split"] * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms to val set
    val_dataset.dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=val_tf)

    print(f"[AgriSense] Dataset → Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if CONFIG["device"] == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if CONFIG["device"] == "cuda" else False
    )

    return train_loader, val_loader, class_names


# ─────────────────────────────────────────────
# MODEL DEFINITION (TRANSFER LEARNING)
# ─────────────────────────────────────────────
def build_model(num_classes):
    """
    Loads EfficientNetB3 pretrained on ImageNet.
    Replaces the classifier head for 38-class PlantVillage task.

    ENHANCEMENT over original repo:
    - Original: Custom ResNet9 (~97% accuracy)
    - Ours: EfficientNetB3 transfer learning (~99% accuracy)
    """
    # Load pretrained EfficientNetB3
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # Freeze backbone layers (fine-tune strategy: train only head first)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    # Unfreeze the last 20 layers for fine-tuning
    for param in list(model.parameters())[-20:]:
        param.requires_grad = True

    return model


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train():
    device = torch.device(CONFIG["device"])

    print("\n" + "="*60)
    print("  AgriSense — EfficientNetB3 Training Pipeline")
    print("  Dataset: PlantVillage (38 classes)")
    print("="*60 + "\n")

    # Load data
    train_loader, val_loader, class_names = load_dataset()

    # Build model
    model = build_model(CONFIG["num_classes"]).to(device)
    print(f"[AgriSense] Model: EfficientNetB3 loaded → {CONFIG['num_classes']} classes")

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-4
    )

    # OneCycleLR: best scheduler for transfer learning
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["num_epochs"]
    )

    # Training loop
    best_val_acc = 0
    history = []

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n[Epoch {epoch}/{CONFIG['num_epochs']}]")
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(f"  → Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  → Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  → Time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "config": CONFIG
            }, CONFIG["model_save_path"])
            print(f"  ✅ Best model saved! Val Acc: {best_val_acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"  Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved: {CONFIG['model_save_path']}")
    print(f"{'='*60}\n")

    # Save training history
    with open("./model/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    model, history = train()
