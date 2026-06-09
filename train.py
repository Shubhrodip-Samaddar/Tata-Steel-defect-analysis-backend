import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import time
import numpy as np

# ─── CONFIG ───────────────────────────────────────────
COMBINED_PATH   = "dataset/combined"
MODEL_SAVE_PATH = "model/defect_model.pth"
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 0.001
IMG_SIZE        = 224
# ──────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomGrayscale(p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomEqualize(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("📂 Loading combined dataset...")
full_dataset = datasets.ImageFolder(root=COMBINED_PATH)
CLASS_NAMES  = full_dataset.classes
NUM_CLASSES  = len(CLASS_NAMES)

print(f"✅ Total: {len(full_dataset)} images")
print(f"📋 Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# Split 80/20 train/val
val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform   = val_transform

# Weighted sampler to handle class imbalance (crease only has 53 images)
targets      = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,   num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,     num_workers=0)

print(f"📊 Train: {train_size} | Val: {val_size}")

# EfficientNet-B0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {device}")
print("🧠 Loading EfficientNet-B0...")

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n🚀 Starting training with {NUM_CLASSES} classes...\n")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_correct, total_train = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        total_train   += labels.size(0)

    model.eval()
    val_correct, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            total_val   += labels.size(0)

    train_acc = 100 * train_correct / total_train
    val_acc   = 100 * val_correct   / total_val
    elapsed   = time.time() - start

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}% | "
          f"Time: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  💾 Best model saved! Val Acc: {val_acc:.1f}%")

    scheduler.step()

# Save class names alongside model
import json
with open("model/class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f)
print(f"\n✅ Done! Best Val Accuracy: {best_val_acc:.1f}%")
print(f"📋 Classes saved to model/class_names.json")
print(f"📁 Model saved to: {MODEL_SAVE_PATH}")