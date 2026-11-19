# inspect_model.py: look inside MobileNetV3-Small and count parameters per part
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Load the network with its ImageNet pre-trained weights
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

# Show the classifier (the "decider" part)
print(model.classifier)

# Count the adjustable numbers (parameters) in one part of the network
def count(part):
    return sum(p.numel() for p in part.parameters())   # numel = how many numbers in a tensor

print("\nParameters per part:")
print("Feature extractor:", count(model.features))
print("Pooling:          ", count(model.avgpool))
print("Classifier:       ", count(model.classifier))
print("Total:            ", count(model))