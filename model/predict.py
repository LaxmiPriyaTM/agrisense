"""
AgriSense - Model Inference Module
====================================
Handles image preprocessing and disease prediction.
Designed to be imported by the Flask API.

Author: SEAI Individual Project
"""

import os
import io
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_b3.pth")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
IMG_SIZE = 224

# ImageNet normalization (matches training transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────
# INFERENCE TRANSFORM
# ─────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# ─────────────────────────────────────────────
# MODEL LOADER (Singleton pattern for Flask)
# ─────────────────────────────────────────────
class ModelManager:
    """
    Singleton class to load model once and reuse across requests.
    This is critical for Flask performance — loading a 50MB model
    per request would be extremely slow.
    """
    _instance = None
    _model = None
    _class_names = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AgriSense Inference] Loading model on {self._device}...")

        # Load class names
        with open(CLASS_NAMES_PATH, "r") as f:
            self._class_names = json.load(f)

        num_classes = len(self._class_names)

        # Rebuild model architecture
        model = models.efficientnet_b3(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # Load saved weights
        checkpoint = torch.load(MODEL_PATH, map_location=self._device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()

        self._model = model
        print(f"[AgriSense Inference] Model loaded! "
              f"Best Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    @property
    def model(self):
        return self._model

    @property
    def class_names(self):
        return self._class_names

    @property
    def device(self):
        return self._device


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────
def validate_image(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Validates uploaded image file.
    Returns (is_valid, error_message)
    """
    # Check file extension
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check file size (max 10MB)
    if len(file_bytes) > 10 * 1024 * 1024:
        return False, "File size exceeds 10MB limit"

    # Try opening as image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Verify it's a valid image
    except Exception:
        return False, "Uploaded file is not a valid image"

    return True, ""


# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict(file_bytes: bytes, top_k: int = 3) -> dict:
    """
    Main inference function called by Flask API.

    Args:
        file_bytes: Raw bytes of the uploaded image
        top_k: Number of top predictions to return

    Returns:
        dict with prediction results and metadata
    """
    manager = ModelManager.get_instance()

    # Preprocess image
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = inference_transform(img).unsqueeze(0).to(manager.device)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

    # Run inference
    with torch.no_grad():
        logits = manager.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Get top-k predictions
    probs, indices = torch.topk(probabilities, k=min(top_k, len(manager.class_names)))
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    # Format results
    top_predictions = []
    for prob, idx in zip(probs, indices):
        class_name = manager.class_names[idx]
        parts = class_name.split("___")
        crop = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ") if len(parts) > 1 else class_name

        top_predictions.append({
            "class_key": class_name,
            "crop": crop,
            "disease": disease,
            "confidence": float(prob * 100),
            "confidence_formatted": f"{prob * 100:.2f}%"
        })

    # Primary prediction (highest confidence)
    primary = top_predictions[0]
    confidence = primary["confidence"]

    # Confidence interpretation
    if confidence >= 90:
        confidence_level = "HIGH"
        confidence_message = "High confidence detection. Treatment recommended."
    elif confidence >= 70:
        confidence_level = "MEDIUM"
        confidence_message = "Moderate confidence. Consider rescanning for confirmation."
    else:
        confidence_level = "LOW"
        confidence_message = "Low confidence. Please retake image with better lighting."

    return {
        "primary_prediction": primary,
        "top_predictions": top_predictions,
        "confidence_level": confidence_level,
        "confidence_message": confidence_message,
        "model_version": "EfficientNetB3-v1.0",
        "num_classes": len(manager.class_names)
    }
