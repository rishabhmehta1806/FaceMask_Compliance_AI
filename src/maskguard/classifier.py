from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

from .device import get_device


class MaskClassifier:
    def __init__(self, checkpoint_path: str | Path, image_size: int = 224, device: str | None = None):
        self.device = device or get_device()
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained classifier checkpoint not found: {path}. "
                "Prepare the labelled dataset and run: python train.py"
            )
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint["class_names"])
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.model = models.mobilenet_v3_large(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        logits = self.model(tensor) / max(self.temperature, 1e-3)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probabilities.astype(np.float32)
