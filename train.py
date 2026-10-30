from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from maskguard.device import get_device

EXPECTED_CLASSES = {"correctly_worn", "nose_exposed", "below_chin", "no_mask"}
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, pretrained: bool):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_logits(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit one scalar temperature on validation logits for better-calibrated confidence."""
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=60)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature.clamp(min=0.05, max=10.0)
        loss = criterion(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.05, max=10.0).item())


def score_logits(logits, targets, class_names, temperature=1.0):
    probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    truth = targets.numpy()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, predictions, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(len(class_names)))).tolist(),
    }


def save_confusion_matrix(matrix, class_names, path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Mask Position Classifier — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the four-class MobileNetV3 mask-position classifier.")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="models/best_mask_classifier.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="Do not start from ImageNet pretrained weights.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"[INFO] Training device: {device}")
    data_root = Path(args.data)
    train_dir, val_dir, test_dir = data_root / "train", data_root / "val", data_root / "test"
    for folder in (train_dir, val_dir, test_dir):
        if not folder.exists():
            raise FileNotFoundError(f"Missing {folder}. First run scripts/prepare_dataset.py")

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.78, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.16, hue=0.025),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), normalize,
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)
    class_names = train_ds.classes
    if set(class_names) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Training requires exactly these four class folders: {sorted(EXPECTED_CLASSES)}. "
            f"Found: {class_names}"
        )
    for name, ds in (("validation", val_ds), ("test", test_ds)):
        if ds.classes != class_names:
            raise RuntimeError(f"{name} class folders/order do not match train split.")

    counts = np.bincount(train_ds.targets, minlength=len(class_names))
    if np.any(counts < 2):
        raise RuntimeError(f"Every training class needs at least 2 samples; found {dict(zip(class_names, counts.tolist()))}")
    print("[INFO] Training samples:", dict(zip(class_names, counts.tolist())))
    class_weights = 1.0 / counts
    sample_weights = [class_weights[label] for label in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(len(class_names), pretrained=not args.scratch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(1) == targets).sum().item())
            total += targets.size(0)
        scheduler.step()

        val_logits, val_targets = get_logits(model, val_loader, device)
        val_metrics = score_logits(val_logits, val_targets, class_names)
        train_loss = total_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} | train_acc={train_accuracy:.4f} "
            f"| val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "temperature": 1.0,
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "architecture": "torchvision.mobilenet_v3_large",
                "image_size": args.image_size,
            }, output_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_logits, val_targets = get_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets)
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, output_path)

    test_logits, test_targets = get_logits(model, test_loader, device)
    test_metrics = score_logits(test_logits, test_targets, class_names, temperature=temperature)
    test_metrics.update({
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "temperature": temperature,
        "class_names": class_names,
        "test_samples": len(test_ds),
        "training_note": "Metrics are generated from the held-out test split; do not compare across different dataset splits without matching protocol.",
    })
    (reports_dir / "evaluation.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(test_metrics["confusion_matrix"], class_names, reports_dir / "confusion_matrix.png")
    print(f"[DONE] Best checkpoint saved to: {output_path.resolve()}")
    print(f"[DONE] Test accuracy={test_metrics['accuracy']:.4f}; macro-F1={test_metrics['macro_f1']:.4f}; temperature={temperature:.3f}")
    print(f"[DONE] Evaluation files saved in: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
