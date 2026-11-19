# extract_features.py: Approach 3 (feature extraction), part 1: extract frozen features
# Turn every image into a 576-number feature vector using the frozen pre-trained network.
# Output: results/features_{train,val,test}_{X,y}.npy
import time
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from datasets import get_loaders

torch.manual_seed(42)                     # fixed seed (reproducibility)

# Pre-trained network, frozen: eval() = inference mode, nothing changes inside
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.eval()

def extract(loader):
    """Pass all images through the feature extractor + pooling, collect the outputs."""
    feats, labels = [], []
    with torch.no_grad():                 # no learning -> no gradients (faster)
        for images, y in loader:
            x = model.features(images)    # conv feature extractor -> [batch, 576, 7, 7]
            x = model.avgpool(x)          # pooling                -> [batch, 576, 1, 1]
            x = torch.flatten(x, 1)       # flatten                -> [batch, 576]
            feats.append(x.numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)

# augment=False: features must come from clean images (no random changes)
train_loader, val_loader, test_loader = get_loaders(batch_size=64, augment=False)

start = time.time()
for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
    X, y = extract(loader)
    np.save(f"results/features_{name}_X.npy", X)   # 576 numbers per image
    np.save(f"results/features_{name}_y.npy", y)   # breed label per image
    print(f"{name}: features {X.shape}, labels {y.shape}")
print(f"Feature extraction time: {time.time() - start:.1f} s")