"""
Train a fine-tuned EfficientNet-B0 for SoCalGuessr.

Uses torchvision's EfficientNet-B0 (~20MB weights) pretrained on ImageNet,
with a replaced classification head for our 6-city task.

Usage:
    python train_vit.py
"""

import pathlib
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DIR = pathlib.Path("./data")

CLASSES = sorted([
    "Anaheim", "Bakersfield", "Los_Angeles",
    "Riverside", "SLO", "San_Diego",
])
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

VALIDATION_FRACTION = 0.2
SEED = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class SoCalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []
        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Model ─────────────────────────────────────────────────────────────────────

class FineTunedModel(nn.Module):
    """EfficientNet-B0 with a replaced classification head."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ── Training logic ────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Data ──────────────────────────────────────────────────────────────
    full_train = SoCalDataset(TRAIN_DIR, transform=train_transform)
    full_val = SoCalDataset(TRAIN_DIR, transform=val_transform)

    n = len(full_train)
    val_size = int(n * VALIDATION_FRACTION)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(range(n), [train_size, val_size], generator=generator)
    train_indices = list(train_indices)
    val_indices = list(val_indices)

    train_subset = torch.utils.data.Subset(full_train, train_indices)
    val_subset = torch.utils.data.Subset(full_val, val_indices)

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train size: {train_size}, Val size: {val_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FineTunedModel(num_classes=NUM_CLASSES)
    lr = 1e-4
    epochs = 15
    weight_decay = 1e-4

    for param in model.backbone.features[:5].parameters():
        param.requires_grad = False

    model = model.to(device)
    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()

        avg_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"loss: {avg_loss:.4f}  "
            f"train_acc: {train_acc:.4f}  "
            f"val_acc: {val_acc:.4f}  "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}  "
            f"[{elapsed:.0f}s]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pt")
            print(f"  -> New best val_acc: {best_val_acc:.4f}, saved model.pt")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    with open("history.json", "w") as f:
        json.dump(history, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Cross-Entropy)")
    ax1.set_title("Training Curve: Loss")
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Curve: Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("Saved training_curve.png")


if __name__ == "__main__":
    main()
