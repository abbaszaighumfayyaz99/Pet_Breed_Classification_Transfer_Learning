# summary.py: one results table for all approaches
# Combines test metrics (evaluate.py), validation results (train.py / train_sklearn.py),
# clean training accuracy (computed here on TRAIN images only), trainable parameters and time.
# Output: results/summary_table.csv
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from datasets import get_loaders

# Training times in seconds, copied from the terminal output of each run
FEATURE_TIME = 84.1   # extract_features.py (shared by both sklearn classifiers)
RUNS = {
    "1_full_finetuning": {"mode": "full", "lr": 1e-4, "epochs": 15, "time_s": 1617},
    "2_head_training":   {"mode": "head", "lr": 1e-3, "epochs": 15, "time_s": 635},
}

def load_cnn(mode):
    """Rebuild the network and load the selected checkpoint."""
    model = mobilenet_v3_small()
    model.classifier[3] = nn.Linear(1024, 37)
    model.load_state_dict(torch.load(f"checkpoints/{mode}_best.pt"))
    model.eval()
    return model

def clean_accuracy(model, loader):
    """Accuracy on clean (non-augmented) images, model in eval mode."""
    correct, n = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            correct += (model(images).argmax(1) == labels).sum().item()
            n += len(labels)
    return correct / n

def count_trainable(model, mode):
    """Full fine-tuning trains everything; head training trains only the classifier."""
    part = model if mode == "full" else model.classifier
    return sum(p.numel() for p in part.parameters())

# Clean TRAIN images (augment=False): only used for the train-vs-test gap
train_loader, _, _ = get_loaders(batch_size=64, augment=False)
test = pd.read_csv("results/test_metrics.csv").set_index("model")   # read, NOT re-evaluated

rows = []

# ---- Approaches 1 and 2 (PyTorch) ----
for name, r in RUNS.items():
    model = load_cnn(r["mode"])
    h = pd.read_csv(f"results/history_{r['mode']}.csv")
    best = h.loc[h["val_acc"].idxmax()]                                # selected epoch
    rows.append({
        "approach": name,
        "selected": f"epoch {int(best['epoch'])}",
        "hyperparameters": f"Adam lr={r['lr']}, batch 32, {r['epochs']} epochs",
        "trainable_params": count_trainable(model, r["mode"]),
        "time_s": r["time_s"],
        "train_acc_clean": clean_accuracy(model, train_loader),
        "val_acc": best["val_acc"],
        "test_acc": test.loc[name, "test_accuracy"],
        "test_macro_f1": test.loc[name, "test_macro_f1"],
    })

# ---- Approach 3 (scikit-learn on frozen features) ----
X_train = np.load("results/features_train_X.npy")
y_train = np.load("results/features_train_y.npy")
sel = pd.read_csv("results/sklearn_selection.csv")
for clf in ["logreg", "svm"]:
    pipe = joblib.load(f"checkpoints/sklearn_{clf}.joblib")
    est = pipe[-1]                                                     # classifier inside the pipeline
    s = sel[sel["classifier"] == clf]
    best = s.loc[s["val_acc"].idxmax()]                                # selected C
    name = f"3_features_{clf}"
    rows.append({
        "approach": name,
        "selected": f"C={best['C']}",
        "hyperparameters": "StandardScaler + " + ("LogisticRegression" if clf == "logreg" else "LinearSVC"),
        "trainable_params": est.coef_.size + est.intercept_.size,      # weights + biases (sklearn)
        "time_s": FEATURE_TIME + best["fit_time_s"],                   # extraction + final fit
        "train_acc_clean": pipe.score(X_train, y_train),
        "val_acc": best["val_acc"],
        "test_acc": test.loc[name, "test_accuracy"],
        "test_macro_f1": test.loc[name, "test_macro_f1"],
    })

table = pd.DataFrame(rows)
table["gap_train_test"] = table["train_acc_clean"] - table["test_acc"]   # overfitting indicator
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
table.to_csv("results/summary_table.csv", index=False)
print("\nSaved: results/summary_table.csv")