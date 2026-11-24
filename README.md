# Homework 2: Image Classification with Transfer Learning

Computer Vision, University of Verona, Fall 2026, Abbas Zaighum Fayyaz

The goal of this homework is to recognize the 37 cat and dog breeds of the **Oxford-IIIT Pet** dataset
using **MobileNetV3-Small**, a small network that was already trained on ImageNet-1K.
The question is how much of that pre-trained network should be trained again. Three options are compared:

1. **Full fine-tuning**: the last layer is replaced and the whole network is trained.
2. **Classifier-head training**: the convolutional part is frozen and only the fully connected
   classifier layers are trained.
3. **Feature extraction + traditional classifier**: the whole network is frozen, each image is turned
   into 576 numbers taken after global average pooling, and scikit-learn classifiers
   (Logistic Regression and a Linear SVM) are trained on those numbers.

All three use the same train/validation/test split, the same preprocessing of the pre-trained weights
(resize, center crop, ImageNet normalization) and the same evaluation. The only difference in the data
pipeline is that approaches 1 and 2 also use random augmentation (random resized crop, horizontal flip)
on the training images, while approach 3 extracts the features once from the original images and reuses them.

## Results (held-out test set, 3,669 images)

| Approach | Selected | Trainable params | Time (s) | Train acc* | Val acc | Test acc | Test macro F1 |
|---|---|---|---|---|---|---|---|
| 1. Full fine-tuning | epoch 12 | 1,555,781 | 1617 | 0.992 | 0.879 | 0.818 | 0.815 |
| 2. Classifier-head training | epoch 3 | 628,773 | 635 | 0.948 | 0.891 | 0.842 | 0.840 |
| 3. Features + Logistic Regression | C = 0.01 | 21,349 | 84.7 | 0.988 | 0.876 | 0.857 | 0.855 |
| 3. Features + Linear SVM | C = 0.01 | 21,349 | 98.9 | 0.999 | 0.875 | 0.833 | 0.831 |

\*Accuracy of the selected model on the training images, without augmentation.
For approaches 1 and 2 the time covers all 15 epochs, including the validation after each epoch.
For approach 3 it is the feature extraction for all splits (84.1 s) plus fitting the selected classifier.
The full table is in `results/summary_table.csv`.

In short: the less the network is retrained, the better it does on new images here. Frozen features with
Logistic Regression give the best test accuracy and are by far the cheapest, while full fine-tuning
overfits the small training set.

Each test accuracy has a 95 % confidence interval of about ±1.2 points. Paired McNemar tests on the
test predictions show that all pairwise differences are significant (p < 0.05), except head training
vs Linear SVM (p = 0.14); the numbers are in `results/significance.csv`. Keep in mind that every approach
was trained only once (one seed, one split).

## Repository structure

```
Pet_Breed_Classification_Transfer_Learning/
├── src/
│   ├── download_data.py     # download the dataset, print split sizes
│   ├── explore_data.py      # images per breed in the official splits
│   ├── make_split.py        # stratified 80/20 train/validation split (seed 42)
│   ├── datasets.py          # shared preprocessing, augmentation and DataLoaders
│   ├── inspect_model.py     # MobileNetV3-Small structure and parameter counts
│   ├── train.py             # approaches 1 and 2 (--mode full / --mode head)
│   ├── extract_features.py  # approach 3, part 1: frozen 576-d features
│   ├── train_sklearn.py     # approach 3, part 2: Logistic Regression + Linear SVM
│   ├── evaluate.py          # single test-set evaluation of all selected models
│   ├── error_analysis.py    # prediction examples and most confused breed pairs
│   ├── plot_curves.py       # training / validation curves
│   ├── summary.py           # final results table
│   └── significance.py      # 95% confidence intervals and McNemar tests
├── splits/                  # train_idx.txt, val_idx.txt (saved split)
├── checkpoints/             # selected models (.pt for PyTorch, .joblib for scikit-learn)
├── results/                 # metrics (.csv), figures (.png), test predictions (.npy)
├── requirements.txt
└── README.md
```

The extracted feature files (`results/features_*.npy`, about 17 MB) are not in the repository.
`src/extract_features.py` creates them again in about 1.5 minutes.

## Environment

| Component | Version |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.2.2 |
| TorchVision | 0.17.2 |
| scikit-learn | 1.6.1 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| Matplotlib | 3.9.4 |

Everything was run on a MacBook Pro (Intel Core i9) on the **CPU only**, without CUDA.
PyTorch 2.2.2 is the last version with wheels for Intel Macs, and it needs NumPy below 2.

To install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

- **Oxford-IIIT Pet Dataset**: 37 breeds (25 dogs, 12 cats), about 200 images per breed.
- Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Authors: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, CVPR 2012.
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0);
  the copyright of the images stays with their owners.

The dataset (about 800 MB) is **not included** here. TorchVision downloads it automatically
into `data/oxford-iiit-pet/` when you run:

```bash
python src/download_data.py
```

The official partitions are `trainval` (3,680 images) and `test` (3,669 images).
The labels used are the 37 breeds, not the binary cat/dog labels.

## How to reproduce

Run all commands from the repository root (the folder containing `src/` and `README.md`), in this order:

```bash
# 1. Data
python src/download_data.py        # download dataset (~800 MB, first run only)
python src/explore_data.py         # images per breed -> results/class_counts_official.csv
python src/make_split.py           # stratified split -> splits/, results/class_counts_split.csv

# 2. Approach 1: full fine-tuning (~27 min on CPU)
python src/train.py --mode full --lr 1e-4 --epochs 15

# 3. Approach 2: classifier-head training (~11 min on CPU)
python src/train.py --mode head --lr 1e-3 --epochs 15

# 4. Approach 3: feature extraction + scikit-learn
python src/extract_features.py     # ~1.5 min
python src/train_sklearn.py        # selects C on validation (~26 min: large C values for the SVM are slow)

# 5. Evaluation
python src/evaluate.py             # test accuracy, macro F1, confusion matrices
python src/error_analysis.py --model 3_features_logreg
python src/error_analysis.py --model 1_full_finetuning
python src/plot_curves.py          # results/training_curves.png
python src/summary.py              # results/summary_table.csv
python src/significance.py         # results/significance.csv
```

`python src/inspect_model.py` is optional; it prints the network structure and the parameter counts.

The training times used in `summary.py` were copied from the terminal output of the runs above.

### Quick check with the included checkpoints (no retraining, about 5-10 min on a CPU, depending on download speed)

If you only want to check the reported results without training everything again, this is enough.
This procedure evaluates the previously trained and saved checkpoints; it does not retrain or modify the models.

```bash
python src/download_data.py        # dataset (~800 MB)
python src/extract_features.py     # recreates results/features_*.npy (~1.5 min)
python src/evaluate.py             # test metrics + confusion matrices from the saved checkpoints
python src/summary.py              # results/summary_table.csv
python src/significance.py         # results/significance.csv
```

`evaluate.py` and `summary.py` need the extracted feature files, which are not stored in the
repository, so `extract_features.py` has to run first.

## Data splits and preprocessing

- The official `trainval` partition is split into **train (2,944)** and **validation (736)** with
  `sklearn.model_selection.train_test_split(test_size=0.2, stratify=breed, random_state=42)`,
  so every breed keeps the same 80/20 proportion. The indices refer to the image order of the official
  `trainval` list as loaded by `torchvision.datasets.OxfordIIITPet`, and they are saved in `splits/`.
- The official `test` partition is kept out of training and model selection. Test images go through
  the fixed, frozen feature extractor (`extract_features.py`) only to produce features for the final
  evaluation. `train_sklearn.py` does not use test features or test labels for fitting the classifiers,
  choosing C or any other decision. Test predictions are made only in the final evaluation
  (`evaluate.py`) and are then used for the error analysis; apart from that, test labels are only used
  to count images per breed (`explore_data.py`). Test-set class counts are reported only as descriptive
  dataset statistics and are not used for training or model selection.
- The classes are almost balanced (93-100 images per breed in trainval, 88-100 in test),
  so no class balancing is applied.
- **Validation, test and feature extraction** use the preprocessing that comes with the pre-trained weights:
  resize to 256 (bilinear), center crop 224, and normalization with the ImageNet
  mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
- **Training (approaches 1 and 2 only)** adds `RandomResizedCrop(224, scale=(0.7, 1.0))` and
  `RandomHorizontalFlip()`, with the same normalization.
- Approach 3 extracts the features once, from images without augmentation.

## Selected models and hyperparameters

Models were selected on the **validation set only**, by best validation accuracy.

| Approach | Trainable layers | Loss / optimizer | Hyperparameters | Selected | Checkpoint |
|---|---|---|---|---|---|
| 1. Full fine-tuning | all layers | Cross-entropy / Adam | lr 1e-4, batch 32, 15 epochs | epoch 12 | `checkpoints/full_best.pt` |
| 2. Head training | `classifier[0]` (Linear 576→1024), `classifier[3]` (Linear 1024→37); `features` frozen (incl. BatchNorm statistics) | Cross-entropy / Adam | lr 1e-3, batch 32, 15 epochs | epoch 3 | `checkpoints/head_best.pt` |
| 3. Logistic Regression | scikit-learn model only | multinomial log-loss, L2 (lbfgs) | StandardScaler (fit on train), C ∈ {0.001, 0.01, 0.1, 1, 10}, max_iter 2000 | C = 0.01 | `checkpoints/sklearn_logreg.joblib` |
| 3. Linear SVM | scikit-learn model only | squared hinge, L2 (liblinear, one-vs-rest) | StandardScaler (fit on train), C ∈ {0.001, 0.01, 0.1, 1, 10}, max_iter 5000 | C = 0.01 | `checkpoints/sklearn_svm.joblib` |

In approaches 1 and 2 the original ImageNet layer (1000 outputs) is replaced by a layer with 37 outputs,
one per breed. Approach 3 does not use the classifier at all; it takes the 576 values that come out of
global average pooling. The pre-trained weights are `torchvision.models.MobileNet_V3_Small_Weights.DEFAULT`
(= `IMAGENET1K_V1`, file `mobilenet_v3_small-047dcff4.pth`, 67.7 % top-1 on ImageNet-1K).

## Reproducibility

- The random seed is **42** everywhere: Python (`random`), NumPy, PyTorch (`torch.manual_seed`),
  the data split (`random_state`), the scikit-learn classifiers and the choice of prediction examples.
- **Deterministic algorithms were not enabled** (`torch.use_deterministic_algorithms` was not used).
  Everything ran on the CPU. Running the first epoch of full fine-tuning a second time gave identical values,
  so the runs look repeatable on this machine, but bit-identical results are not guaranteed.
- Different hardware, a different number of CPU threads or other library versions can still change
  the results slightly.
- No test data was used for training, early stopping, feature scaling or choosing hyperparameters.

## References

- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* CVPR 2012.
- A. Howard et al. *Searching for MobileNetV3.* ICCV 2019.
- PyTorch / TorchVision: https://pytorch.org, scikit-learn: https://scikit-learn.org
