# datasets.py: shared data loading for all three approaches
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

# Official preprocessing that belongs to the pre-trained weights:
# resize to 256 -> center crop 224 -> convert to tensor -> normalize (ImageNet mean/std)
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
EVAL_TF = WEIGHTS.transforms()
MEAN, STD = EVAL_TF.mean, EVAL_TF.std

# Training: same normalization + random augmentation (training set only)
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # random zoom/crop
    transforms.RandomHorizontalFlip(),                    # random mirror
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders(batch_size=32, augment=True):
    """Return train, val and test DataLoaders using the saved split."""
    train_tf = TRAIN_TF if augment else EVAL_TF

    # Two copies of trainval: one with training transforms, one with clean transforms
    train_full = OxfordIIITPet("data", split="trainval", target_types="category", transform=train_tf)
    val_full = OxfordIIITPet("data", split="trainval", target_types="category", transform=EVAL_TF)
    test_ds = OxfordIIITPet("data", split="test", target_types="category", transform=EVAL_TF)

    # Load the saved split and keep only those images
    train_idx = np.loadtxt("splits/train_idx.txt", dtype=int)
    val_idx = np.loadtxt("splits/val_idx.txt", dtype=int)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    # DataLoader = gives images in batches; shuffle only for training
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Quick test: runs only when this file is executed directly
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader))
    print("Batch images:", images.shape)
    print("Batch labels:", labels[:8])
    print("Batches -> train:", len(train_loader), "val:", len(val_loader), "test:", len(test_loader))