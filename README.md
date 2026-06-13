# CNN AHD Model — CIFAR-10 Image Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x41hd/Cnn_Ahd_Model/blob/main/cnn_ahd.ipynb)

A Convolutional Neural Network (CNN) trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, achieving ~84% validation accuracy. Built and trained in Google Colab with a T4 GPU.

---

## Overview

This project implements and iteratively improves a CNN for 10-class image classification. Two model versions are explored:

- **V1 (Baseline):** Simple CNN — 4 Conv2D blocks + Dense layers. Reaches ~77% val accuracy but overfits heavily after epoch 7.
- **V2 (Improved):** Adds `Dropout` and `BatchNormalization` to combat overfitting. Reaches **~84% val accuracy** with stable training.

The final model is saved as `cnn_Ahd_v1.h5` and evaluated using a confusion matrix and random prediction samples.

---

## Dataset — CIFAR-10

| Property | Value |
|---|---|
| Training samples | 50,000 |
| Test samples | 10,000 |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | 10 |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Model Architecture (V2)

```
Input (32×32×3)
  → Conv2D(32, 3×3, relu) → Dropout(0.2) → BatchNorm
  → Conv2D(64, 3×3, relu) → MaxPool(2) → Dropout(0.2) → BatchNorm
  → Conv2D(64, 3×3, relu) → MaxPool(2) → Dropout(0.2) → BatchNorm
  → Conv2D(128, 3×3, relu) → MaxPool(2) → Dropout(0.2)
  → Flatten → Dropout(0.2) → BatchNorm
  → Dense(128, relu) → Dropout(0.2) → BatchNorm
  → Dense(10, softmax)
```

**Total trainable parameters:** ~398K

---

## Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 30 |
| Batch size | 64 |
| Input normalization | Pixel values divided by 255 |
| Labels | One-hot encoded |

---

## Results

| Metric | V1 (Baseline) | V2 (Regularized) |
|---|---|---|
| Best val accuracy | ~77.3% | ~84.0% |
| Overfitting | Severe | Minimal |

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install via:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

Open the notebook directly in Google Colab using the badge at the top, or clone and run locally:

```bash
git clone https://github.com/0x41hd/Cnn_Ahd_Model.git
cd Cnn_Ahd_Model
jupyter notebook cnn_ahd.ipynb
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
