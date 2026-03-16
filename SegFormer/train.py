import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from models.segformer_model import SegFormerChange
from utils.dataset import ChangeDataset
from utils.metrics import update_confusion_matrix, compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size, img_size):
    train_ds = ChangeDataset(
        config.get_split_dir("train"),
        img_size=img_size,
        num_classes=config.NUM_CLASSES,
        label_values=config.LABEL_VALUES,
    )
    val_ds = ChangeDataset(
        config.get_split_dir("val"),
        img_size=img_size,
        num_classes=config.NUM_CLASSES,
        label_values=config.LABEL_VALUES,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and config.DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and config.DEVICE == "cuda",
    )
    return train_loader, val_loader


def compute_class_weights(dataset, num_classes):
    class_counts = np.zeros(num_classes, dtype=np.float64)
    labels_dir = os.path.join(dataset.root, "labels")

    for name in dataset.files:
        label_path = os.path.join(labels_dir, name)
        label = cv2.imread(label_path, 0)
        if label is None:
            continue

        if dataset.label_lut is not None:
            label = dataset.label_lut[label]

        bincount = np.bincount(label.reshape(-1), minlength=num_classes)
        class_counts += bincount[:num_classes]

    class_counts = np.maximum(class_counts, 1.0)
    median_count = np.median(class_counts)
    weights = median_count / class_counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(imgs)
        outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(outputs, labels, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader):
    model.eval()
    conf_matrix = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.long)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            conf_matrix = update_confusion_matrix(conf_matrix, labels.cpu(), preds.cpu(), config.NUM_CLASSES)

    return compute_metrics(conf_matrix)


def save_checkpoint(path, model, optimizer, epoch, metrics):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--save_dir", default=str(config.SAVE_DIR))
    parser.add_argument("--disable_class_weights", action="store_true")
    args = parser.parse_args()

    set_seed(config.SEED)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "segformer_change_best.pt"
    last_path = save_dir / "segformer_change_last.pt"

    train_loader, val_loader = build_loaders(args.batch_size, args.img_size)

    model = SegFormerChange(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_weights = None
    if not args.disable_class_weights:
        class_weights = compute_class_weights(train_loader.dataset, config.NUM_CLASSES).to(config.DEVICE)
        print("class_weights:", class_weights.detach().cpu().numpy().round(4).tolist())

    best_miou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights=class_weights)
        val_metrics = evaluate(model, val_loader)
        epoch_time_sec = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_miou": val_metrics["miou"],
            "epoch_time_sec": round(epoch_time_sec, 2),
        }
        history.append(record)

        save_checkpoint(last_path, model, optimizer, epoch, val_metrics)
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(best_path, model, optimizer, epoch, val_metrics)

        print(json.dumps(record, indent=2))

    hist_path = save_dir / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()