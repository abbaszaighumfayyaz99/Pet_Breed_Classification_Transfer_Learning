# train.py: train MobileNetV3-Small on the 37 breeds
#   Approach 1 (full fine-tuning):        python src/train.py --mode full --lr 1e-4 --epochs 15
#   Approach 2 (classifier-head training): python src/train.py --mode head --lr 1e-3 --epochs 15

import argparse, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from datasets import get_loaders

# ---------------- Settings from the command line ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["full", "head"], required=True)  # which approach
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-4)                   # learning rate = step size
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# ---------------- Reproducibility: fix all random seeds ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- Data ----------------
# Train (with augmentation) + validation. The test loader is ignored on purpose:
# the test set is NEVER used during training or model selection.
train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size, augment=True)

# ---------------- Model ----------------
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)  # pre-trained on ImageNet
model.classifier[3] = nn.Linear(1024, 37)    # replace final layer: 1000 -> 37 outputs

if args.mode == "head":
    for p in model.features.parameters():
        p.requires_grad = False              # freeze the feature extractor (Approach 2)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Mode: {args.mode} | trainable parameters: {trainable}")

# ---------------- Loss and optimizer ----------------
criterion = nn.CrossEntropyLoss()            # standard loss for classification
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr)  # updates trainable params only

# ---------------- One pass over a dataset ----------------
def run_epoch(loader, train):
    model.train() if train else model.eval()
    if args.mode == "head":
        model.features.eval()                # keep frozen part fully frozen (BatchNorm stats too)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):      # gradients only when training
        for images, labels in loader:
            outputs = model(images)                   # 1. forward: 37 scores per image
            loss = criterion(outputs, labels)         # 2. loss: how wrong
            if train:
                optimizer.zero_grad()                 #    clear old gradients
                loss.backward()                       # 3. backward: compute gradients
                optimizer.step()                      # 4. step: update parameters
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()  # highest score = prediction
            n += len(labels)
    return total_loss / n, correct / n      # average loss, accuracy

# ---------------- Training loop ----------------
history, best_val_acc, best_epoch = [], 0.0, 0
start = time.time()
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
          f"| val loss {val_loss:.3f} acc {val_acc:.3f}")

    # Model selection: keep the epoch with the best VALIDATION accuracy
    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch
        torch.save(model.state_dict(), f"checkpoints/{args.mode}_best.pt")
        print("   -> best so far, checkpoint saved")

train_time = time.time() - start
print(f"\nBest epoch: {best_epoch} (val acc {best_val_acc:.3f}) | training time: {train_time:.0f} s")

# Save the training history for plots and the report
pd.DataFrame(history).to_csv(f"results/history_{args.mode}.csv", index=False)