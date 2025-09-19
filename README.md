# Facial Expression Recognition with Vision Transformers (FER2013)

Robust Facial Expression Recognition (FER) using a pretrained **ViT/MAE** backbone with partial freezing and fine-tuning on **FER2013**.

---

## Overview

- **Backbone:** Vision Transformer Masked Autoencoder (**ViTMAE**) pretrained on ImageNet
- **Transfer learning:** Freeze **embeddings + blocks [0..3]**, fine-tune higher layers
- **Task:** 7-class FER on FER2013 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Frameworks:** PyTorch, Hugging Face `transformers`, `timm`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data: FER2013

Download 'Dataset/archive.zip'  

Typical transforms:
- Resize to **224×224**
- Random crop + horizontal flip
- Normalize to ImageNet stats

---

## Quickstart

Open and run:
```
Code Files/FER_ViT.ipynb
```

The notebook:
- Loads FER2013
- Builds a **ViTMAE** classifier head
- **Freezes embeddings + first 4 transformer blocks**
- Trains with **AdamW** + **CosineAnnealing**
- Logs validation accuracy and loss
- Saves plots

---

## Training Configuration

- **Backbone:** `ViTMAEModel` (Hugging Face)
- **Head:** Linear classifier on `[CLS]` token → 7 classes
- **Frozen:** embeddings + blocks `[0..3]` (total 12 blocks)
- **Loss:** CrossEntropyLoss
- **Optimizer:** AdamW (lr=3e-5, weight decay)
- **Scheduler:** Cosine Annealing
- **Augmentations:** resize(224), random crop, horizontal flip, normalization

---

## Results (from latest run)

- **Epoch 1:** loss=**1.7723**, val_acc=**14.81%**
- **Epoch 2:** loss=**1.5935**, val_acc=**16.27%**
- **Epoch 3:** loss=**1.4681**, val_acc=**25.85%**
- **Epoch 4:** loss=**1.4131**, val_acc=**35.64%**

**Observations**
- Outperforms random guessing (~14.3%) early on
- Stronger recognition for **Happy**/**Surprise**
- Confusions between **Fear** and **Disgust** (class imbalance effect)

---

## Evaluation

The notebook evaluates on validation/test split and can produce:
- Accuracy
- Classification report
- Confusion matrix

---

## Authors

- **Hila Levi Yosefi**
- **Noa Amsalem**
