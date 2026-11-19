# explore_data.py: count images per breed in the official splits
# Output: table in the terminal + results/class_counts_official.csv

from collections import Counter                   # counts repeated values
import pandas as pd                               # tables
from torchvision.datasets import OxfordIIITPet    # dataset loader

# Load both official splits (already downloaded)
trainval = OxfordIIITPet(root="data", split="trainval", target_types="category")
test = OxfordIIITPet(root="data", split="test", target_types="category")

# Count images per breed number (fast: reads labels only, not images)
trainval_counts = Counter(trainval._labels)
test_counts = Counter(test._labels)

# One row per breed: name + counts
rows = []
for idx, name in enumerate(trainval.classes):
    rows.append({"breed": name, "trainval": trainval_counts[idx], "test": test_counts[idx]})
table = pd.DataFrame(rows)

# Show results
print(table.to_string(index=False))
print("\nTotal:", table["trainval"].sum(), "trainval,", table["test"].sum(), "test")
print("Min/Max per breed (trainval):", table["trainval"].min(), "/", table["trainval"].max())

# Save for the report
table.to_csv("results/class_counts_official.csv", index=False)
print("Saved to results/class_counts_official.csv")