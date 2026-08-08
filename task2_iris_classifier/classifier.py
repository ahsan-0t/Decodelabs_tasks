"""
Project 2 - Data Classification Using AI
DecodeLabs AI/ML Industrial Training Kit (Batch 2026)
Author: Ahsan Tahir

Description:
A supervised learning pipeline that classifies Iris flowers into one of
three species (Setosa, Versicolor, Virginica) using the K-Nearest
Neighbors (KNN) algorithm. Demonstrates the full IPO (Input -> Process ->
Output) workflow: loading data, scaling features, splitting into
train/test sets, training a model, and evaluating it.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# PHASE 1: INPUT - Load and understand the dataset
# ---------------------------------------------------------------------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris


def explore_data(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} samples, {df.shape[1] - 1} features, 3 classes\n")
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Class distribution:")
    print(df["species"].value_counts(), "\n")
    print("Basic statistics:")
    print(df.describe(), "\n")


# ---------------------------------------------------------------------
# PHASE 2: PROCESS - Scale, split, train
# ---------------------------------------------------------------------
def prepare_data(df, iris):
    X = df[iris.feature_names]
    y = iris.target  # numeric labels, aligned with iris.target_names

    # Train/test split (80/20), stratified to keep class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling: KNN is distance-based, so features must be on the
    # same scale (StandardScaler -> mean 0, variance 1).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def find_best_k(X_train, y_train, X_test, y_test, max_k=15):
    """
    Try different K values and track error rate to find the 'elbow'.
    K=1 often looks best on a tiny test set but tends to overfit (it just
    memorizes the single nearest point), so among K values tied for the
    lowest error, we prefer the largest (more stable/general) K.
    """
    errors = []
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors.append(1 - accuracy_score(y_test, preds))

    min_error = min(errors)
    tied_ks = [k for k, err in enumerate(errors, start=1) if err == min_error]
    best_k = max(tied_ks)  # prefer the larger, more stable K among ties
    return best_k, errors


def train_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# PHASE 3: OUTPUT - Evaluate and validate
# ---------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(report)

    return predictions, acc, f1, cm


def plot_confusion_matrix(cm, target_names, out_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names
    )
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.title("Confusion Matrix - KNN Iris Classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {out_path}")


def plot_k_error_curve(errors, best_k, out_path="k_error_curve.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker="o", linestyle="dashed",
             color="steelblue", markerfacecolor="orange")
    plt.axvline(best_k, color="red", linestyle=":", label=f"Best K = {best_k}")
    plt.title("Error Rate vs. K Value (Choosing the Best K)")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"K error curve plot saved to: {out_path}")


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    df, iris = load_data()
    explore_data(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df, iris)

    best_k, errors = find_best_k(X_train, y_train, X_test, y_test)
    print(f"Best K found: {best_k} (lowest error rate on test set)\n")
    plot_k_error_curve(errors, best_k)

    model = train_model(X_train, y_train, best_k)

    predictions, acc, f1, cm = evaluate_model(
        model, X_test, y_test, iris.target_names
    )
    plot_confusion_matrix(cm, iris.target_names)

    # Demonstrate prediction on a brand-new, unseen flower sample
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
    sample_scaled = scaler.transform(sample)
    sample_pred = model.predict(sample_scaled)
    print(f"\nSample prediction for measurements {sample.iloc[0].tolist()}: "
          f"{iris.target_names[sample_pred[0]]}")


if __name__ == "__main__":
    main()
