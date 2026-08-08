# Project 2 — Data Classification Using AI (Iris Species Classifier)

**DecodeLabs AI/ML Industrial Training Kit (Batch 2026)**
**Author:** Ahsan Tahir

## 📌 Overview
This project builds a supervised machine learning model that classifies
Iris flowers into one of three species — **Setosa**, **Versicolor**, or
**Virginica** — based on four physical measurements: sepal length, sepal
width, petal length, and petal width. It uses the classic Iris dataset
(built into scikit-learn) and the **K-Nearest Neighbors (KNN)** algorithm.

## 🎯 Goal
Build a basic classification model that:
- Loads and explores a real dataset
- Splits the data into training and testing sets
- Applies a simple classification algorithm (KNN)
- Evaluates performance using proper metrics — not just accuracy

## ⚙️ Pipeline (Input → Process → Output)
1. **Input** — Load the Iris dataset (150 samples, 4 features, 3 balanced classes of 50 each).
2. **Process**
   - Split data into 80% training / 20% testing using a stratified split, so all 3 species stay balanced in both sets.
   - Scale features using `StandardScaler` (mean = 0, variance = 1) — necessary because KNN is a distance-based algorithm and unscaled features would bias it.
   - Test K values from 1 to 15 and select the one with the lowest test error (preferring a larger, more stable K when there's a tie, since K=1 tends to overfit by just memorizing the nearest point).
   - Train a `KNeighborsClassifier` using the chosen K.
3. **Output**
   - Predict species on the held-out test set.
   - Evaluate using **accuracy**, **macro F1 score**, a **confusion matrix**, and a full **classification report**.
   - Demonstrate a prediction on one brand-new, unseen flower sample.

## 📊 Results
| Metric | Value |
|---|---|
| Best K | 15 |
| Accuracy | 96.7% |
| F1 Score (macro) | 0.967 |

Only 1 misclassification out of 30 test samples — a Versicolor flower predicted as Virginica (these two species are known to slightly overlap in petal measurements).

## ▶️ How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python3 classifier.py
```
This prints the dataset overview and model evaluation to the console, and saves two plots:
- `k_error_curve.png` — error rate vs. K value, used to justify the chosen K
- `confusion_matrix.png` — per-class prediction accuracy

## 📂 Files in This Repository
- `classifier.py` — full pipeline: load → scale → split → train → evaluate
- `confusion_matrix.png` — output visualization
- `k_error_curve.png` — output visualization
- `README.md` — this file

## 🚀 Key Skills Demonstrated
Data handling with pandas, feature scaling, train/test splitting,
supervised learning with K-Nearest Neighbors, hyperparameter tuning
(choosing K), and model evaluation using multiple metrics (accuracy,
precision, recall, F1, confusion matrix) rather than accuracy alone.
