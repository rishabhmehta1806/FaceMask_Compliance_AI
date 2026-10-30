from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from maskguard.classifier import MaskClassifier
from maskguard.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on a labelled image folder.")
    parser.add_argument("--data", default="data/processed/test")
    parser.add_argument("--checkpoint", default="models/best_mask_classifier.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if dataset.classes != class_names:
        raise RuntimeError(f"Folder classes {dataset.classes} do not match checkpoint classes {class_names}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    classifier = MaskClassifier(checkpoint_path, image_size=image_size)
    all_probs, all_true = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = classifier.model(images.to(classifier.device)) / max(classifier.temperature, 1e-3)
            all_probs.extend(torch.softmax(logits, 1).cpu().numpy())
            all_true.extend(targets.numpy())
    predictions = np.asarray(all_probs).argmax(1)
    report = {
        "accuracy": float(accuracy_score(all_true, predictions)),
        "macro_f1": float(f1_score(all_true, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(all_true, predictions, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(all_true, predictions, labels=list(range(len(class_names)))).tolist(),
        "class_names": class_names,
        "samples": len(dataset),
    }
    out = ROOT / "reports" / "external_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("accuracy", "macro_f1", "samples", "class_names")}, indent=2))
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
