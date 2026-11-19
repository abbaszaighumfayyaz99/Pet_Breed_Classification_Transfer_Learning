# plot_curves.py: training curves (train vs validation) for Approaches 1 and 2
# Reads results/history_full.csv and results/history_head.csv (saved by train.py)
# Output: results/training_curves.png
import pandas as pd
import matplotlib.pyplot as plt

runs = [("full", "Approach 1: full fine-tuning", "tab:blue"),
        ("head", "Approach 2: head training", "tab:orange")]

fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.5))
for mode, label, color in runs:
    h = pd.read_csv(f"results/history_{mode}.csv")
    best = h["val_acc"].idxmax()                         # row of the selected (best) epoch

    # Accuracy: dashed = train, solid = validation
    ax_acc.plot(h["epoch"], h["train_acc"], "--", color=color, label=f"{label} (train)")
    ax_acc.plot(h["epoch"], h["val_acc"], "-", color=color, label=f"{label} (val)")
    ax_acc.plot(h["epoch"][best], h["val_acc"][best], "o", color=color, markersize=9)  # selected epoch

    # Loss: same style
    ax_loss.plot(h["epoch"], h["train_loss"], "--", color=color, label=f"{label} (train)")
    ax_loss.plot(h["epoch"], h["val_loss"], "-", color=color, label=f"{label} (val)")

ax_acc.set(title="Accuracy (dot = selected epoch)", xlabel="Epoch", ylabel="Accuracy")
ax_loss.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
for ax in (ax_acc, ax_loss):
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
print("Saved: results/training_curves.png")