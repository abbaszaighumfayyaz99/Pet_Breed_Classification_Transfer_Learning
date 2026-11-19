# train_sklearn.py: Approach 3 (feature extraction), part 2: train scikit-learn classifiers
# Tries Logistic Regression and Linear SVM; chooses C for each on the VALIDATION set.
# Output: checkpoints/sklearn_logreg.joblib, checkpoints/sklearn_svm.joblib,
#         results/sklearn_selection.csv
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Load saved features (test features are NOT loaded: used only in the final evaluation)
X_train = np.load("results/features_train_X.npy")
y_train = np.load("results/features_train_y.npy")
X_val = np.load("results/features_val_X.npy")
y_val = np.load("results/features_val_y.npy")

# The two classifiers to compare; C is chosen on validation
classifiers = {
    "logreg": lambda C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
    "svm":    lambda C: LinearSVC(C=C, max_iter=5000, random_state=SEED),
}
C_values = [0.001, 0.01, 0.1, 1, 10]

rows = []
for name, make_clf in classifiers.items():
    best = None
    for C in C_values:
        # Pipeline: the scaler is fitted on TRAIN data only, then the classifier is trained
        model = make_pipeline(StandardScaler(), make_clf(C))
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        train_acc = model.score(X_train, y_train)   # accuracy on training features
        val_acc = model.score(X_val, y_val)         # accuracy on validation features
        rows.append({"classifier": name, "C": C, "train_acc": train_acc,
                     "val_acc": val_acc, "fit_time_s": fit_time})
        print(f"{name:6s} C={C:<6} | train acc {train_acc:.3f} | val acc {val_acc:.3f} | fit {fit_time:.1f} s")

        # Model selection: keep the C with the best validation accuracy
        if best is None or val_acc > best["val_acc"]:
            best = {"C": C, "val_acc": val_acc, "model": model}

    joblib.dump(best["model"], f"checkpoints/sklearn_{name}.joblib")   # save the chosen model
    print(f"  -> best {name}: C={best['C']} (val acc {best['val_acc']:.3f}), saved\n")

pd.DataFrame(rows).to_csv("results/sklearn_selection.csv", index=False)