# error_analysis.py: prediction examples and most confused breed pairs
# Usage: python src/error_analysis.py --model 3_features_logreg
# Output: results/examples_<model>.png, results/top_confusions_<model>.csv
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="3_features_logreg")   # which model's predictions
args = parser.parse_args()

# Original test photos (no transform -> we see the real image, not the 224x224 tensor)
test = OxfordIIITPet(root="data", split="test", target_types="category")
names = test.classes
y_true = np.array(test._labels)
y_pred = np.load(f"results/test_preds_{args.model}.npy")      # saved by evaluate.py

# Pick 6 random correct and 6 random wrong predictions (fixed seed -> same picks every run)
rng = np.random.default_rng(42)
correct = rng.choice(np.where(y_true == y_pred)[0], 6, replace=False)
wrong = rng.choice(np.where(y_true != y_pred)[0], 6, replace=False)

# Figure: top row = correct (green), bottom row = wrong (red)
fig, axes = plt.subplots(2, 6, figsize=(18, 7))
for row, idxs, color in [(0, correct, "green"), (1, wrong, "red")]:
    for ax, i in zip(axes[row], idxs):
        img, _ = test[i]                                     # load photo number i
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"True: {names[y_true[i]]}\nPred: {names[y_pred[i]]}",
                     color=color, fontsize=9)
fig.suptitle(f"Test predictions: {args.model} (top: correct, bottom: wrong)")
plt.tight_layout()
plt.savefig(f"results/examples_{args.model}.png", dpi=150)

# Most frequent mistakes: count every (true, predicted) pair where they differ
pairs = Counter((names[t], names[p]) for t, p in zip(y_true, y_pred) if t != p)
top = pd.DataFrame([{"true": t, "predicted": p, "count": c}
                    for (t, p), c in pairs.most_common(10)])
print(top.to_string(index=False))
top.to_csv(f"results/top_confusions_{args.model}.csv", index=False)
print(f"Saved: results/examples_{args.model}.png + top_confusions_{args.model}.csv")