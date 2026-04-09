# Handwritten Digit Recognition App

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)


[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Team](#-team)

</div>

---

## Overview

This project demonstrates an end-to-end handwritten digit recognition system that combines classical machine learning techniques to achieve efficient and accurate predictions without relying on deep learning. The system uses **Principal Component Analysis (PCA)** for dimensionality reduction and **Random Forest** for classification.

<blockquote>This project features:</blockquote>

-  **Fast Predictions**: Real-time digit recognition
-  **High Accuracy**: Trained on 70,000 MNIST images
-  **Dimensionality Reduction**: 784 → 80 features using PCA
-  **Ensemble Learning**: 300 decision trees for robust predictions
-  **User-Friendly GUI**: Built with PyQt5
-  **Confidence Scores**: Probability distribution for all digits (0-9)

---

## Features
-  Recognize single or multiple handwritten digits from images
-  Automatic digit segmentation and preprocessing
-  Center-of-mass alignment for better accuracy
-  Confidence score visualization with bar charts
-  Support for various image formats (PNG, JPG, JPEG)
-  File browser for easy image selection
-  Display of preprocessed digit images
-  Real-time confidence scores for each prediction
-  Clear prediction output with average confidence

---

##  Installation
You need to have:
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: First Run (Model Training)
On first run of ```GUI.py```, the application will:
1. Download the MNIST dataset (~55 MB)
2. Train the Random Forest model
3. Save the model as `digit_rf_mnist.pkl`
**Note**: This process takes a looong time.

### Running the GUI Application

```bash
python GUI.py
```

### Using the Application

1. **Browse**: Click the "Browse" button to select an image file
2. **Predict**: Click the "Predict" button to run recognition
3. **View Results**: 
   - Processed digits appear in the left panel
   - Confidence scores are displayed in the right panel
   - A popup shows the final prediction
---

## How It Works

### Pipeline Overview

```
Input Image → Preprocessing → Feature Extraction → Classification → Output
```

### 1️⃣ **Preprocessing**

The input image undergoes several transformations to match the MNIST dataset format:

- **Grayscale Conversion**: Convert RGB to single-channel
- **Gaussian Blur**: Reduce noise with 3×3 kernel
- **Otsu's Thresholding**: Automatic binarization
- **Inversion Check**: Ensure white digits on black background
- **Morphological Operations**: Closing and dilation for cleanup
- **Contour Detection**: Automatically segment individual digits
- **Resize & Pad**: Scale to 20×20, then pad to 28×28
- **Center-of-Mass Alignment**: Shift digit to center

### 2️⃣ **Feature Extraction (PCA)**

```
Original: 784 features (28×28 pixels)
    ↓
  PCA Transformation
    ↓
Reduced: 80 principal components (89.8% variance retained)
```

**Why PCA?**
- Reduces computational complexity
- Removes redundant features
- Speeds up training and prediction
- Prevents overfitting

### 3️⃣ **Classification (Random Forest)**

```
300 Decision Trees → Majority Voting → Final Prediction
```

**Model Configuration:**
- `n_estimators=300`: Number of trees
- `random_state=42`: Reproducibility
- `n_jobs=-1`: Parallel processing

**Output:**
- Predicted digit (0-9)
- Confidence score (%)
- Probability distribution for all classes

---

## Model Architecture

### Dataset: MNIST

- **Source**: [OpenML MNIST](https://www.openml.org/d/554)
- **Size**: 70,000 images (60,000 train + 10,000 test)
- **Format**: 28×28 grayscale images
- **Classes**: 10 digits (0-9)
- **Pixel Range**: 0 (black) to 255 (white)

### Dimensionality Reduction

| Stage | Dimensions | Description |
|-------|-----------|-------------|
| Raw Image | 28×28 = 784 | Original pixel values |
| Normalized | 784 | Scaled to [0, 1] |
| PCA Transform | 80 | Principal components |

### Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=300,      # 300 decision trees
    random_state=42,       # Reproducible results
    n_jobs=-1             # Use all CPU cores
)
```
---

## 🔧 Preprocessing Pipeline

### Detailed Steps

```python
1. Load Image
   ↓
2. Convert to Grayscale
   ↓
3. Apply Gaussian Blur (3×3)
   ↓
4. Otsu's Automatic Thresholding
   ↓
5. Invert if Needed (white digit on black)
   ↓
6. Morphological Closing (2×2 kernel)
   ↓
7. Dilation (1 iteration)
   ↓
8. Find Contours
   ↓
9. Filter Small Contours (area < 50px)
   ↓
10. Sort Contours Left-to-Right
   ↓
11. For Each Contour:
    • Extract bounding box
    • Add 4px padding
    • Apply Gaussian blur
    • Resize to 20×20
    • Pad to 28×28
    • Center-of-mass alignment
   ↓
12. Normalize to [0, 1]
   ↓
13. Ready for PCA + Classification
```

### Center-of-Mass Alignment

```python
# Calculate center of mass
cx = Σ(x * intensity) / Σ(intensity)
cy = Σ(y * intensity) / Σ(intensity)

# Shift to center (14, 14)
shift_x = 14 - cx
shift_y = 14 - cy

# Apply affine transformation
```

-> This ensures digits are centered like MNIST training data.
---

## 👥 Team

This project was developed as a mini-project for Linear Algebra course.

| Name | SRN |
|------|-------------|
| Pranav Chandrasekar | CS332 |
| Pranav S P | CS337 |
| Pranav S S | CS336 |
| Rakshan Pandian | CS364 |

Hope we get full marks!
