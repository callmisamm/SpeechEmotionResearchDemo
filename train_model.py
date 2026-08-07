"""
============================================================
Speech Emotion Recognition
CNN Training Script
Part 1 : Load Features + Build CNN
============================================================
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

print("=" * 60)
print("Speech Emotion Recognition")
print("CNN Model Training")
print("=" * 60)

# ==========================================================
# CREATE MODELS FOLDER
# ==========================================================

os.makedirs("models", exist_ok=True)

# ==========================================================
# LOAD FEATURES
# ==========================================================

print("\nLoading Saved Features...")

X_train = np.load("saved_features/X_train.npy")
y_train = np.load("saved_features/y_train.npy")

X_val = np.load("saved_features/X_val.npy")
y_val = np.load("saved_features/y_val.npy")

X_test = np.load("saved_features/X_test.npy")
y_test = np.load("saved_features/y_test.npy")

print("\nFeature Shapes")

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

print("\nLabel Shapes")

print("y_train :", y_train.shape)
print("y_val   :", y_val.shape)
print("y_test  :", y_test.shape)

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

label_encoder = joblib.load(
    "saved_features/label_encoder.pkl"
)

class_names = label_encoder.classes_

NUM_CLASSES = len(class_names)

print("\nClasses")

for i, c in enumerate(class_names):
    print(f"{i} : {c}")

print("\nBuilding Improved CNN...")

model = Sequential([

    Input(shape=(128,128,1)),

    ########################################################
    # Block 1
    ########################################################

    Conv2D(
        32,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.25),

    ########################################################
    # Block 2
    ########################################################

    Conv2D(
        64,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.30),

    ########################################################
    # Block 3
    ########################################################

    Conv2D(
        128,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    Conv2D(
        128,
        (3,3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),
    Dropout(0.35),

    ########################################################

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    Dropout(0.50),

    Dense(NUM_CLASSES, activation="softmax")

])

# ==========================================================
# MODEL SUMMARY
# ==========================================================

print()

model.summary()

# ==========================================================
# COMPILE MODEL
# ==========================================================

print("\nCompiling Model...")

optimizer = Adam(
    learning_rate=3e-4
)

model.compile(

    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=["accuracy"]

)

print("\nModel Compiled Successfully!")

print("=" * 60)
# ==========================================================
# CALLBACKS
# ==========================================================

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

print("\nCreating Callbacks...")

checkpoint = ModelCheckpoint(
    filepath="models/cnn_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(
    "results/training_log.csv",
    append=False
)

callbacks = [
    checkpoint,
    early_stop,
    reduce_lr,
    csv_logger
]

print("Callbacks Ready.")

# ==========================================================
# CLASS WEIGHTS
# ==========================================================

print("\nComputing Class Weights...")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

print(class_weights)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n" + "=" * 60)
print("Starting CNN Training")
print("=" * 60)

EPOCHS = 100
BATCH_SIZE = 64

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

print("\nTraining Completed Successfully!")

# ==========================================================
# EVALUATE MODEL
# ==========================================================

print("\n" + "=" * 60)
print("Evaluating Model")
print("=" * 60)

train_loss, train_accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Results")
print("-" * 40)

print(f"Training Accuracy   : {train_accuracy*100:.2f}%")
print(f"Validation Accuracy : {val_accuracy*100:.2f}%")
print(f"Testing Accuracy    : {test_accuracy*100:.2f}%")

print()

print(f"Training Loss       : {train_loss:.4f}")
print(f"Validation Loss     : {val_loss:.4f}")
print(f"Testing Loss        : {test_loss:.4f}")

print("=" * 60)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

print("\nSaving Final Model...")

model.save(
    "models/final_cnn_model.keras"
)

print("Final model saved.")

# ==========================================================
# SAVE HISTORY
# ==========================================================

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "results/history.csv",
    index=False
)

print("Training history saved.")

# ==========================================================
# BEST EPOCH
# ==========================================================

best_epoch = np.argmax(
    history.history["val_accuracy"]
)

best_val_acc = np.max(
    history.history["val_accuracy"]
)

best_val_loss = np.min(
    history.history["val_loss"]
)

print("\nBest Epoch :", best_epoch + 1)

print(
    "Best Validation Accuracy : {:.2f}%".format(
        best_val_acc * 100
    )
)

print(
    "Best Validation Loss : {:.4f}".format(
        best_val_loss
    )
)

print("=" * 60)

print("\nTraining Finished Successfully.")

print("\nSaved Files")
print("-------------------------------")
print("models/cnn_model.keras")
print("models/final_cnn_model.keras")
print("results/history.csv")
print("results/training_log.csv")

print("=" * 60)