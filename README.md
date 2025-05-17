# Herbify: Deep Learning Framework for Precise Herb Identification

## Overview

Herbify is a comprehensive deep learning project focused on the classification of 91 types of herbs using advanced computer vision techniques. The project includes data preprocessing, augmentation, model training (with various state-of-the-art architectures), hyperparameter optimization, and ensemble learning.

---

## Directory Structure & Contents

### 1. `Preprocessing/`

- **Augmentation/**: Jupyter notebooks for image augmentation using Albumentations and OpenCV. Includes multiple strategies (A, B, C) for increasing dataset diversity.

- **Data_Split/**: Notebook for splitting the dataset into train, test, and validation sets (default: 70/15/15 split).

- **PAHD/**: Stands for Preprocessing Algorithm for Herb (object) Detection. Notebooks for advanced preprocessing, including background removal, contour extraction, and adaptive histogram equalization. Includes single-image and batch processing examples.

### 2. `Optimal-Hyperparameters/`

- **Optimal_Hyperparameters_Grid_Search.ipynb**: Grid search for optimal batch size, learning rate, and weight decay. Includes training, validation, and test evaluation for each configuration.

### 3. `Fine-Tuning/`

Contains subfolders for each model family, each with a notebook for transfer learning and fine-tuning on the herb dataset:

- **EfficientNet/**: `EfficientNet_v2_Large.ipynb`
- **MobileNet/**: `MobileNet_v3_Large.ipynb`
- **ResNet/**: `ResNet152.ipynb`
- **VGG/**: `VGG16.ipynb`, `VGG19.ipynb`
- **ViT/**: Vision Transformer models (`ViT_B_16_e2e.ipynb`, `ViT_B_16_v1.ipynb`, `ViT_L_16_e2e.ipynb`, `ViT_L_16_v1.ipynb`)

Each notebook covers:

- Data loading and preprocessing
- Model setup and architecture (along with parameter freezing, early stopping and other configurations)
- Transfer learning (or Fine-tuning)
- Training, validation, and evaluation (accuracy, F1, precision, recall, confusion matrix)

### 4. `Ensemble/`

Jupyter notebooks for combining predictions from multiple models to improve classification performance. Includes various ensemble strategies:

- MobileNet v3 Large + ViT-B/16
- VGG19 + ViT-B/16

- ResNet152 + ViT-L/16
- EfficientNet v2 Large + ViT-L/16

- MobileNet v3 Large + VGG19 + ViT-B/16
- ResNet152 + EfficientNet v2 Large + ViT-L/16

- MobileNet v3 Large + VGG19 + EfficientNet v2 Large + ViT-L/16
- VGG19 + ResNet152 + EfficientNet v2 Large + ViT-L/16

Multi-model ensembles (e.g., ResNet, EfficientNet, VGG19, ViT)

---

## Getting Started

1. **Requirements**: Python 3.8+, PyTorch, torchvision, scikit-learn, albumentations, OpenCV, matplotlib, tqdm, colorama, PIL, numpy, pandas, scikit-image.

2. **Data**: Place your herb image dataset in the appropriate directory as referenced in the notebooks.

3. **Run Notebooks**: Follow the order: Preprocessing → Hyperparameter Search (for best models) → Fine-Tuning → Ensemble.

---

## Project Highlights

- **Flexible Preprocessing**: Advanced background removal and augmentation.
- **Multiple Architectures**: Fine-tuning of CNNs (VGG, ResNet, EfficientNet, MobileNet) and Transformers (ViT).
- **Ensemble Learning**: Combine strengths of different models.
- **Reproducibility**: Fixed random seeds and clear data splits.

---

## Herb Datasets

- **DIMPSAR**: [DIMPSAR Dataset](https://data.mendeley.com/datasets/748f8jkphb/2)
- **DeepHerb**: [DeepHerb Dataset](https://data.mendeley.com/datasets/nnytj2v3n5/1)
- **Herbify**: Custom dataset of 91 herbs, available on request.

---

## Citation

If you use this project, please cite appropriately.

---

## License

This project is for academic and research purposes licensed under the MIT License. For commercial use, please contact the author.
