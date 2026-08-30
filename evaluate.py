"""
============================================================
Speech Emotion Recognition
Model Evaluation Script
============================================================
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

# ==========================================================
# CREATE RESULTS FOLDER
# ==========================================================

os.makedirs("results", exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading Trained CNN Model")
print("=" * 60)

model = load_model("models/cnn_model_v2.keras")

print("Model Loaded Successfully!")

# ==========================================================
# LOAD TEST DATA
# ==========================================================

X_test = np.load("saved_features/X_test.npy")
y_test = np.load("saved_features/y_test.npy")

label_encoder = joblib.load(
    "saved_features/label_encoder.pkl"
)

class_names = label_encoder.classes_

print("\nTest Samples :", len(X_test))
print("Classes      :", list(class_names))

# ==========================================================
# MODEL EVALUATION
# ==========================================================

print("\nEvaluating Model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nGenerating Predictions...")

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# ==========================================================
# OVERALL ACCURACY
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nOverall Accuracy : {:.2f}%".format(
    accuracy * 100
))

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix\n")
print(cm)

# ==========================================================
# SAVE CONFUSION MATRIX FIGURE
# ==========================================================

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved.")

# ==========================================================
# SAVE NUMERIC MATRIX
# ==========================================================

np.savetxt(
    "results/confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print("\nSaved Files")
print("-----------------------------")
print("results/classification_report.txt")
print("results/confusion_matrix.png")
print("results/confusion_matrix.csv")

print("=" * 60)