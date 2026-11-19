# make_split.py: split trainval into train (80%) + val (20%), stratified by breed
# Output: splits/train_idx.txt, splits/val_idx.txt, results/class_counts_split.csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

SEED = 42  # fixed seed -> the same split every run (reproducibility)

trainval = OxfordIIITPet(root="data", split="trainval", target_types="category")
labels = np.array(trainval._labels)     # breed number of each image
indices = np.arange(len(labels))        # image positions: 0, 1, ..., 3679

# stratify=labels -> every breed keeps the same 80/20 ratio
train_idx, val_idx = train_test_split(indices, test_size=0.2,
                                      stratify=labels, random_state=SEED)

# Save the split, so every model uses exactly the same images
np.savetxt("splits/train_idx.txt", np.sort(train_idx), fmt="%d")
np.savetxt("splits/val_idx.txt", np.sort(val_idx), fmt="%d")

# Images per breed in train and val (for the report)
counts = pd.DataFrame({
    "breed": trainval.classes,
    "train": np.bincount(labels[train_idx], minlength=37),
    "val": np.bincount(labels[val_idx], minlength=37),
})
print(counts.to_string(index=False))
print("\nTrain:", len(train_idx), "| Val:", len(val_idx))
counts.to_csv("results/class_counts_split.csv", index=False)