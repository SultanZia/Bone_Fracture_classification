# Bone Fracture Detection from X-ray Images

**Deep Learning Assessment — MSc Data Science, Manchester Metropolitan University**

Binary classification of X-ray images into **Fractured** / **Not Fractured** using a custom CNN and VGG16 transfer learning, with Grad-CAM visualisation for clinical interpretability.

---

## Key Results

| Model | Test Accuracy | Test AUC | Epochs |
|-------|--------------|----------|--------|
| Custom CNN | **99%** | 0.992 | 10 |
| VGG16 (Transfer Learning) | **99%** | 0.993 | 3 |

**Key finding:** VGG16 achieves equivalent accuracy in 3 epochs vs 10 for the custom CNN, demonstrating the efficiency of transfer learning. Grad-CAM heatmaps confirm both models focus on anatomically relevant bone regions when predicting fractures.

---

## Dataset

**Bone Fracture Detection Dataset** — Kaggle  
🔗 [https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)

| Split | Images |
|-------|--------|
| Train | 13,000 |
| Test  | 4,000  |
| **Total** | **17,000** |

**Classes:** Fractured · Not Fractured  
**Image type:** X-ray radiographs (resized to 224×224)

---

## Methodology

### Custom CNN
- 3 convolutional blocks: Conv2D → BatchNormalization → MaxPool2D → Dropout(0.3)
- Filters: 32 → 64 → 128
- Classification head: Dense(256) → Dropout → Dense(128) → Dropout → Dense(1, sigmoid)
- Loss: Binary cross-entropy | Optimiser: Adam
- Callbacks: EarlyStopping (patience=5), ModelCheckpoint

### VGG16 Transfer Learning
- Frozen VGG16 backbone (ImageNet weights)
- Custom head: Flatten → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)
- Converges in 3 epochs — significantly more efficient than training from scratch

### Grad-CAM Visualisation
- Gradient-weighted Class Activation Mapping highlights regions driving each prediction
- Applied to both models using their respective final convolutional layers
- Confirms models attend to bone structure rather than image artefacts — critical for clinical trustworthiness

---

## Repository Structure

```
bone-fracture-detection/
├── bone_fracture_detection.ipynb   ← clean notebook with Grad-CAM
├── train.py                        ← training script (CNN + VGG16)
├── predict.py                      ← single-image inference with Grad-CAM
├── requirements.txt
├── .gitignore
└── data/
    └── README.md                   ← dataset download instructions
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x, Keras |
| Computer Vision | OpenCV, Grad-CAM |
| Data Handling | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Environment | Google Colab (T4 GPU) |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/SultanZia/bone-fracture-detection.git
cd bone-fracture-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
See `data/README.md` for instructions. Update `DATA_DIR` in the notebook config cell.

### 4. Train models
```bash
python train.py --model cnn --epochs 10
python train.py --model vgg16 --epochs 3
```

### 5. Run inference on a single X-ray
```bash
python predict.py --model_path cnn_best_model.keras --image_path xray.jpg
```

---

## Clinical Relevance

Undiagnosed bone fractures are a significant clinical problem — misdiagnosis leads to delayed or incorrect treatment. This project demonstrates that deep learning models can achieve 99% accuracy on X-ray fracture detection, with Grad-CAM providing the interpretability needed for clinicians to trust and validate model predictions.

---

## Author

**Mohammed Zia Sultan**  
MSc Data Science, Manchester Metropolitan University (2023–2024)  
[github.com/SultanZia](https://github.com/SultanZia)
