# evaluate.py: evaluate the selected models once on the held-out test set
# Output: results/test_metrics.csv, results/confusion_<model>.png, results/test_preds_<model>.npy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import mobilenet_v3_small
from datasets import get_loaders

# Test images with clean preprocessing; shuffle=False keeps a fixed order
_, _, test_loader = get_loaders(batch_size=64, augment=False)
class_names = test_loader.dataset.classes                 # 37 breed names
y_true = np.array(test_loader.dataset._labels)            # true breed of each test image

def load_cnn(mode):
    """Rebuild the network and load our trained weights (best checkpoint)."""
    model = mobilenet_v3_small()                          # architecture only
    model.classifier[3] = nn.Linear(1024, 37)             # same 37-output layer as in training
    model.load_state_dict(torch.load(f"checkpoints/{mode}_best.pt"))
    model.eval()                                          # inference mode
    return model

def predict_cnn(model):
    """Predicted breed for every test image."""
    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            preds.append(model(images).argmax(1).numpy())  # highest score = prediction
    return np.concatenate(preds)

# ---- Predictions of all selected models ----
preds = {
    "1_full_finetuning": predict_cnn(load_cnn("full")),
    "2_head_training":   predict_cnn(load_cnn("head")),
}
X_test = np.load("results/features_test_X.npy")
assert (np.load("results/features_test_y.npy") == y_true).all()   # same image order check
for clf in ["logreg", "svm"]:
    model = joblib.load(f"checkpoints/sklearn_{clf}.joblib")
    preds[f"3_features_{clf}"] = model.predict(X_test)

# ---- Metrics + confusion matrix for each model ----
rows = []
for name, y_pred in preds.items():
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")        # every breed counts equally
    rows.append({"model": name, "test_accuracy": acc, "test_macro_f1": f1})
    print(f"{name:20s} | test acc {acc:.3f} | macro F1 {f1:.3f}")
    np.save(f"results/test_preds_{name}.npy", y_pred)     # for prediction examples later

    cm = confusion_matrix(y_true, y_pred)                 # rows = true, columns = predicted
    fig, ax = plt.subplots(figsize=(16, 16))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False, text_kw={"fontsize": 6})
    ax.set_title(f"Confusion matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{name}.png", dpi=150)
    plt.close(fig)

pd.DataFrame(rows).to_csv("results/test_metrics.csv", index=False)
print("Saved: results/test_metrics.csv + confusion matrix images")