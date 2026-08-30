"""
============================================================
Speech Emotion Recognition
Improved CNN Training
============================================================

Purpose:
    Train a stronger CNN while keeping the existing
    preprocessing, dataset and deployment interface.

Important:
    This script DOES NOT overwrite the current deployed model.

New model files:
    models/cnn_model_new.keras
    models/final_cnn_model_new.keras
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    GlobalAveragePooling2D
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("Speech Emotion Recognition")
print("IMPROVED CNN TRAINING")
print("=" * 70)

print("\nTensorFlow Version:", tf.__version__)

# ==========================================================
# LOAD FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("Loading Saved Features")
print("=" * 70)

X_train = np.load(
    "saved_features/X_train.npy"
)

y_train = np.load(
    "saved_features/y_train.npy"
)

X_val = np.load(
    "saved_features/X_val.npy"
)

y_val = np.load(
    "saved_features/y_val.npy"
)

X_test = np.load(
    "saved_features/X_test.npy"
)

y_test = np.load(
    "saved_features/y_test.npy"
)

print("\nFeature Shapes")

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_val   :", X_val.shape)
print("y_val   :", y_val.shape)

print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

# ==========================================================
# BASIC VALIDATION
# ==========================================================

print("\nChecking feature values...")

print(
    "Training range:",
    X_train.min(),
    "to",
    X_train.max()
)

print(
    "Validation range:",
    X_val.min(),
    "to",
    X_val.max()
)

print(
    "Testing range:",
    X_test.min(),
    "to",
    X_test.max()
)

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

label_encoder = joblib.load(
    "saved_features/label_encoder.pkl"
)

class_names = label_encoder.classes_

NUM_CLASSES = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nNumber of classes:", NUM_CLASSES)

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("Class Distribution")
print("=" * 70)

train_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

val_counts = np.bincount(
    y_val,
    minlength=NUM_CLASSES
)

test_counts = np.bincount(
    y_test,
    minlength=NUM_CLASSES
)

print("\nTraining:")

for i, name in enumerate(class_names):
    print(
        f"{name:10s}: {train_counts[i]}"
    )

print("\nValidation:")

for i, name in enumerate(class_names):
    print(
        f"{name:10s}: {val_counts[i]}"
    )

print("\nTesting:")

for i, name in enumerate(class_names):
    print(
        f"{name:10s}: {test_counts[i]}"
    )

# ==========================================================
# CLASS WEIGHTS
# ==========================================================

print("\n" + "=" * 70)
print("Computing Class Weights")
print("=" * 70)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(i): float(w)
    for i, w in enumerate(weights)
}

print("\nClass Weights:")

for i, name in enumerate(class_names):
    print(
        f"{name:10s}: {class_weights[i]:.4f}"
    )

# ==========================================================
# BUILD IMPROVED CNN
# ==========================================================

print("\n" + "=" * 70)
print("Building Improved CNN")
print("=" * 70)

model = Sequential([

    Input(
        shape=(128, 128, 1)
    ),

    # ======================================================
    # BLOCK 1
    # ======================================================

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(
        0.20
    ),

    # ======================================================
    # BLOCK 2
    # ======================================================

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(
        0.25
    ),

    # ======================================================
    # BLOCK 3
    # ======================================================

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(
        0.30
    ),

    # ======================================================
    # BLOCK 4
    # ======================================================

    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(
        0.35
    ),

    # ======================================================
    # CLASSIFICATION HEAD
    # ======================================================

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Dropout(
        0.40
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==========================================================
# MODEL SUMMARY
# ==========================================================

print()

model.summary()

# ==========================================================
# COMPILE
# ==========================================================

print("\n" + "=" * 70)
print("Compiling Model")
print("=" * 70)

optimizer = Adam(
    learning_rate=1e-4
)

model.compile(

    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

print("\nModel compiled successfully.")

# ==========================================================
# CALLBACKS
# ==========================================================

print("\n" + "=" * 70)
print("Creating Training Callbacks")
print("=" * 70)

checkpoint = ModelCheckpoint(

    filepath="models/cnn_model_new.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1
)

csv_logger = CSVLogger(

    "results/improved_training_log.csv",

    append=False
)

callbacks = [

    checkpoint,
    early_stop,
    reduce_lr,
    csv_logger

]

print("Callbacks ready.")

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

EPOCHS = 80

BATCH_SIZE = 32

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("Epochs       :", EPOCHS)
print("Batch Size   :", BATCH_SIZE)
print("Learning Rate:", 1e-4)
print("Seed         :", SEED)

# ==========================================================
# TRAIN
# ==========================================================

print("\n" + "=" * 70)
print("STARTING IMPROVED CNN TRAINING")
print("=" * 70)

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    class_weight=class_weights,

    shuffle=True,

    verbose=1
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

print("\nSaving final improved model...")

model.save(
    "models/final_cnn_model_new.keras"
)

print(
    "Saved:",
    "models/final_cnn_model_new.keras"
)

# ==========================================================
# LOAD BEST CHECKPOINT
# ==========================================================

print("\nLoading best validation checkpoint...")

best_model = tf.keras.models.load_model(
    "models/cnn_model_new.keras"
)

print("Best model loaded successfully.")

# ==========================================================
# EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

train_loss, train_accuracy = best_model.evaluate(
    X_train,
    y_train,
    verbose=0
)

val_loss, val_accuracy = best_model.evaluate(
    X_val,
    y_val,
    verbose=0
)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nResults")
print("-" * 50)

print(
    f"Training Accuracy   : {train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy : {val_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy    : {test_accuracy * 100:.2f}%"
)

print()

print(
    f"Training Loss       : {train_loss:.4f}"
)

print(
    f"Validation Loss     : {val_loss:.4f}"
)

print(
    f"Testing Loss        : {test_loss:.4f}"
)

# ==========================================================
# SAVE HISTORY
# ==========================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "results/improved_history.csv",
    index=False
)

print(
    "\nTraining history saved:"
)

print(
    "results/improved_history.csv"
)

# ==========================================================
# BEST EPOCH
# ==========================================================

best_epoch = np.argmax(
    history.history["val_accuracy"]
)

best_val_accuracy = np.max(
    history.history["val_accuracy"]
)

best_val_loss = np.min(
    history.history["val_loss"]
)

print("\n" + "=" * 70)
print("BEST TRAINING RESULT")
print("=" * 70)

print(
    "Best Epoch:",
    best_epoch + 1
)

print(
    "Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Lowest Validation Loss: "
    f"{best_val_loss:.4f}"
)

# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary = {

    "best_epoch":
        int(best_epoch + 1),

    "best_validation_accuracy":
        float(best_val_accuracy),

    "best_validation_accuracy_percent":
        float(best_val_accuracy * 100),

    "training_accuracy":
        float(train_accuracy),

    "validation_accuracy":
        float(val_accuracy),

    "test_accuracy":
        float(test_accuracy),

    "training_loss":
        float(train_loss),

    "validation_loss":
        float(val_loss),

    "test_loss":
        float(test_loss)

}

pd.DataFrame(
    [summary]
).to_csv(
    "results/improved_model_summary.csv",
    index=False
)

# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("IMPROVED CNN TRAINING FINISHED")
print("=" * 70)

print("\nNew Files:")

print(
    "models/cnn_model_new.keras"
)

print(
    "models/final_cnn_model_new.keras"
)

print(
    "results/improved_history.csv"
)

print(
    "results/improved_training_log.csv"
)

print(
    "results/improved_model_summary.csv"
)

print("\nIMPORTANT:")
print(
    "The original models/cnn_model.keras was NOT modified."
)

print("=" * 70)