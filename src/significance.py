# significance.py: 95% confidence intervals of test accuracy and paired McNemar tests between approaches
# Reads results/test_preds_<model>.npy (saved by evaluate.py) and the official test labels.
# Output: results/significance.csv
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from torchvision.datasets import OxfordIIITPet

MODELS = ["1_full_finetuning", "2_head_training", "3_features_logreg", "3_features_svm"]

y_true = np.array(OxfordIIITPet(root="data", split="test", target_types="category")._labels)
correct = {m: np.load(f"results/test_preds_{m}.npy") == y_true for m in MODELS}
n = len(y_true)

# 95% confidence interval of each test accuracy (normal approximation of a binomial)
for m in MODELS:
    acc = correct[m].mean()
    half = 1.96 * math.sqrt(acc * (1 - acc) / n)
    print(f"{m:20s} test acc {acc:.3f}  95% CI [{acc - half:.3f}, {acc + half:.3f}]")

# McNemar test: only images where exactly one of the two models is right are informative
rows = []
for a, b in combinations(MODELS, 2):
    only_a = int((correct[a] & ~correct[b]).sum())
    only_b = int((~correct[a] & correct[b]).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue      # exact test
    rows.append({"model_a": a, "model_b": b, "only_a_correct": only_a,
                 "only_b_correct": only_b, "p_value": p})
    print(f"{a} vs {b}: only A right {only_a}, only B right {only_b}, p = {p:.2g}")

pd.DataFrame(rows).to_csv("results/significance.csv", index=False)
print("Saved: results/significance.csv")