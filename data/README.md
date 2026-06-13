# Data Setup

This project uses the **Bone Fracture Detection Dataset** from Kaggle.

## Download

1. Go to: **https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project**
2. Click **Download** (requires a free Kaggle account)
3. Extract the zip file

## Expected Structure

```
data/
├── train/
│   ├── fractured/
│   │   ├── image1.jpg
│   │   └── ...
│   └── not fractured/
│       └── ...
├── val/
│   ├── fractured/
│   └── not fractured/
└── test/
    ├── fractured/
    └── not fractured/
```

## Update Config

Once extracted, update `DATA_DIR` in the notebook's Section 1 config cell:

```python
# Local
DATA_DIR = './data'

# Google Colab
DATA_DIR = '/content/drive/MyDrive/datasetBone'
```

## Dataset Stats

| Split | Images |
|-------|--------|
| Train | ~13,000 |
| Test  | ~4,000 |
| **Total** | **~17,000** |

Classes: `fractured` · `not fractured`
