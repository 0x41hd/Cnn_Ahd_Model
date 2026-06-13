# CNN Image Classifier — CIFAR-10

A Convolutional Neural Network (CNN) built with Keras/TensorFlow to classify images from the CIFAR-10 dataset into 10 categories. The project demonstrates the impact of regularization techniques (Dropout + Batch Normalization) on reducing overfitting.

---

## Dataset

**CIFAR-10** — 60,000 color images (32×32 px) across 10 classes:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

- 50,000 training images
- 10,000 test images

---

## Model Architecture

Two models were trained and compared:

### Model v1 — Baseline (without regularization)
```
Conv2D(32) → Conv2D(64) → MaxPool →
Conv2D(64) → MaxPool →
Conv2D(128) → MaxPool →
Flatten → Dense(128) → Dense(10, softmax)
```
- **Result:** ~97% train accuracy / ~75% test accuracy → **severe overfitting**

### Model v2 — Regularized (final model saved as `cnn_Ahd_v1.h5`)
```
Conv2D(32) → Dropout(0.2) → BatchNorm →
Conv2D(64) → MaxPool → Dropout(0.2) → BatchNorm →
Conv2D(64) → MaxPool → Dropout(0.2) → BatchNorm →
Conv2D(128) → MaxPool → Dropout(0.2) →
Flatten → Dropout(0.2) → BatchNorm →
Dense(128) → Dropout(0.2) → BatchNorm →
Dense(10, softmax)
```
- **Result:** ~85% train accuracy / ~84% test accuracy → **much better generalization**

---

## Results

| Model | Train Accuracy | Test Accuracy | Overfitting |
|-------|---------------|--------------|-------------|
| Baseline | ~97% | ~75% | Severe |
| Regularized | ~85% | ~84% | Minimal |

Adding **Dropout** and **Batch Normalization** reduced the train/test gap from ~22% down to ~1%.

---

## Setup

```bash
# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# Open the notebook in Google Colab or Jupyter
jupyter notebook cnn_ahd.ipynb
```

> **Recommended:** Run on Google Colab with GPU (T4) for faster training.

---

## Usage

```python
from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("cnn_Ahd_v1.h5")

# Predict on new images (normalized to 0-1 range)
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)  # add batch dimension

predictions = model.predict(image)
predicted_class = labels[np.argmax(predictions)]
```

**Classes:**
```python
labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
```

---

## Key Takeaways

- **Batch Normalization** stabilizes training and speeds up convergence
- **Dropout** prevents co-adaptation of neurons and reduces overfitting
- Both techniques together allow the model to generalize well on unseen data
- Without regularization, the model memorizes the training data instead of learning patterns

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 64 |
| Input Shape | (32, 32, 3) |

---

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x41hd/Cnn_Ahd_Model/blob/main/cnn_ahd.ipynb)
