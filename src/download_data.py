# download_data.py: download the Oxford-IIIT Pet dataset and print its size
# Source: https://www.robots.ox.ac.uk/~vgg/data/pets/  |  License: CC BY-SA 4.0

from torchvision.datasets import OxfordIIITPet

for split in ["trainval", "test"]:
    ds = OxfordIIITPet(root="data", split=split, target_types="category", download=True)
    print(split, ":", len(ds), "images")

print("Number of breeds:", len(ds.classes))
print("First 5 breeds:", ds.classes[:5])