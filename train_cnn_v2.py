


# ==========================================================
# IMPORTS
# ==========================================================

import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("Speech Emotion Recognition")
print("CNN Training - Version 2")
print("=" * 70)

print("\nCurrent Directory:")
print(os.getcwd())


# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================================
# FILE PATHS
# ==========================================================

X_TRAIN_PATH = "saved_features/X_train.npy"
Y_TRAIN_PATH = "saved_features/y_train.npy"

X_VAL_PATH = "saved_features/X_val.npy"
Y_VAL_PATH = "saved_features/y_val.npy"

X_TEST_PATH = "saved_features/X_test.npy"
Y_TEST_PATH = "saved_features/y_test.npy"

LABEL_ENCODER_PATH = "saved_features/label_encoder.pkl"


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = [
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_VAL_PATH,
    Y_VAL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    LABEL_ENCODER_PATH
]

print("\nChecking required files...")

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print("FOUND:", file_path)

print("\nAll required files found.")


# ==========================================================
# LOAD SAVED FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("Loading Saved Features")
print("=" * 70)

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


# ==========================================================
# DISPLAY SHAPES
# ==========================================================

print("\nFeature Shapes")

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_val   :", X_val.shape)
print("y_val   :", y_val.shape)

print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

print("\nLoading Label Encoder...")

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

class_names = label_encoder.classes_

NUM_CLASSES = len(class_names)

print("\nClasses:")

for i, class_name in enumerate(class_names):

    print(
        f"{i} : {class_name}"
    )

print("\nNumber of Classes:", NUM_CLASSES)


# ==========================================================
# VERIFY INPUT SHAPE
# ==========================================================

EXPECTED_SHAPE = (128, 128, 1)

if X_train.shape[1:] != EXPECTED_SHAPE:

    raise ValueError(
        "\nUnexpected input shape.\n"
        f"Expected: {EXPECTED_SHAPE}\n"
        f"Found: {X_train.shape[1:]}"
    )

print(
    "\nInput shape verified:",
    EXPECTED_SHAPE
)


# ==========================================================
# DATA DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("Training Label Distribution")
print("=" * 70)

unique_train, counts_train = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(
    unique_train,
    counts_train
):

    print(
        f"{class_names[label]:<12} : {count}"
    )


print("\nValidation Label Distribution")

unique_val, counts_val = np.unique(
    y_val,
    return_counts=True
)

for label, count in zip(
    unique_val,
    counts_val
):

    print(
        f"{class_names[label]:<12} : {count}"
    )


print("\nTest Label Distribution")

unique_test, counts_test = np.unique(
    y_test,
    return_counts=True
)

for label, count in zip(
    unique_test,
    counts_test
):

    print(
        f"{class_names[label]:<12} : {count}"
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================
#
# The training data is already augmented.
#
# We calculate balanced weights directly from y_train.
#
# This helps prevent the model from completely ignoring
# minority / difficult classes.
# ==========================================================

print("\n" + "=" * 70)
print("Computing Class Weights")
print("=" * 70)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(classes, weights)
}

for class_id, weight in class_weights.items():

    print(
        f"{class_names[class_id]:<12} : {weight:.4f}"
    )


# ==========================================================
# BUILD CNN
# ==========================================================

print("\n" + "=" * 70)
print("Building CNN Version 2")
print("=" * 70)


model = Sequential([

    # ======================================================
    # INPUT
    # ======================================================

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
        0.25
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
        0.30
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
        0.35
    ),


    # ======================================================
    # GLOBAL AVERAGE POOLING
    # ======================================================

    GlobalAveragePooling2D(),


    # ======================================================
    # DENSE LAYER
    # ======================================================

    Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),

    Dropout(
        0.50
    ),


    # ======================================================
    # OUTPUT
    # ======================================================

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ==========================================================
# MODEL SUMMARY
# ==========================================================

print("\n")

model.summary()


# ==========================================================
# COMPILE MODEL
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

print(
    "Initial Learning Rate:",
    1e-4
)


# ==========================================================
# CALLBACK PATHS
# ==========================================================

BEST_MODEL_PATH = (
    "models/cnn_model_v2.keras"
)

FINAL_MODEL_PATH = (
    "models/final_cnn_model_v2.keras"
)

HISTORY_PATH = (
    "results/history_v2.csv"
)

TRAINING_LOG_PATH = (
    "results/training_log_v2.csv"
)


# ==========================================================
# CALLBACKS
# ==========================================================

print("\n" + "=" * 70)
print("Creating Training Callbacks")
print("=" * 70)


checkpoint = ModelCheckpoint(

    filepath=BEST_MODEL_PATH,

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

    TRAINING_LOG_PATH,

    append=False

)


callbacks = [

    checkpoint,

    early_stop,

    reduce_lr,

    csv_logger

]


print("\nCallbacks ready.")


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

EPOCHS = 80

BATCH_SIZE = 32


print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    "Maximum Epochs :", EPOCHS
)

print(
    "Batch Size      :", BATCH_SIZE
)

print(
    "Initial LR      :", 1e-4
)

print(
    "Best Model      :", BEST_MODEL_PATH
)

print(
    "Final Model     :", FINAL_MODEL_PATH
)


# ==========================================================
# START TRAINING
# ==========================================================

print("\n" + "=" * 70)
print("Starting CNN Version 2 Training")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "The existing CNN models will NOT be overwritten."
)

print(
    "You can safely stop training with Ctrl+C."
)

print(
    "The best completed checkpoint will remain saved."
)

print("\n")


try:

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

except KeyboardInterrupt:

    print("\n")
    print("=" * 70)
    print("TRAINING INTERRUPTED BY USER")
    print("=" * 70)

    print(
        "\nTraining was stopped with Ctrl+C."
    )

    print(
        "The latest completed best checkpoint remains:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nDo NOT delete the model."
    )

    print(
        "You can evaluate the saved checkpoint separately."
    )

    raise SystemExit


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

print("\n" + "=" * 70)
print("Saving Training History")
print("=" * 70)


history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


print(
    "Training history saved:"
)

print(
    HISTORY_PATH
)


# ==========================================================
# BEST EPOCH
# ==========================================================

best_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    )
)

best_val_accuracy = float(
    np.max(
        history.history["val_accuracy"]
    )
)

best_val_loss = float(
    np.min(
        history.history["val_loss"]
    )
)


print("\n" + "=" * 70)
print("BEST TRAINING RESULT")
print("=" * 70)

print(
    f"Best Epoch              : {best_epoch + 1}"
)

print(
    "Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Loss     : {best_val_loss:.4f}"
)


# ==========================================================
# EVALUATE TRAINING SET
# ==========================================================

print("\n" + "=" * 70)
print("Evaluating Training Set")
print("=" * 70)


train_loss, train_accuracy = model.evaluate(

    X_train,
    y_train,

    verbose=1

)


# ==========================================================
# EVALUATE VALIDATION SET
# ==========================================================

print("\n" + "=" * 70)
print("Evaluating Validation Set")
print("=" * 70)


val_loss, val_accuracy = model.evaluate(

    X_val,
    y_val,

    verbose=1

)


# ==========================================================
# EVALUATE TEST SET
# ==========================================================

print("\n" + "=" * 70)
print("Evaluating Test Set")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test,

    verbose=1

)


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("FINAL CNN VERSION 2 RESULTS")
print("=" * 70)


print(
    f"\nTraining Accuracy   : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy    : "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"\nTraining Loss       : "
    f"{train_loss:.4f}"
)

print(
    f"Validation Loss     : "
    f"{val_loss:.4f}"
)

print(
    f"Testing Loss        : "
    f"{test_loss:.4f}"
)


# ==========================================================
# GENERATE TEST PREDICTIONS
# ==========================================================

print("\n" + "=" * 70)
print("Generating Test Predictions")
print("=" * 70)


y_prob = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ==========================================================
# OVERALL ACCURACY
# ==========================================================

overall_accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nOverall Test Accuracy : "
    f"{overall_accuracy * 100:.2f}%"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_test,

    y_pred,

    target_names=class_names,

    digits=4,

    zero_division=0

)


print(report)


REPORT_PATH = (
    "results/classification_report_v2.txt"
)


with open(
    REPORT_PATH,
    "w"
) as f:

    f.write(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(

    y_test,

    y_pred

)


print(cm)


# ==========================================================
# SAVE CONFUSION MATRIX CSV
# ==========================================================

CM_CSV_PATH = (
    "results/confusion_matrix_v2.csv"
)


np.savetxt(

    CM_CSV_PATH,

    cm,

    delimiter=",",

    fmt="%d"

)


# ==========================================================
# SAVE CONFUSION MATRIX IMAGE
# ==========================================================

CM_IMAGE_PATH = (
    "results/confusion_matrix_v2.png"
)


plt.figure(
    figsize=(10, 8)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)


plt.title(
    "CNN Version 2 - Confusion Matrix"
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)


plt.tight_layout()


plt.savefig(

    CM_IMAGE_PATH,

    dpi=300

)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    CM_IMAGE_PATH
)


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

print("\n" + "=" * 70)
print("Saving Final Model")
print("=" * 70)


model.save(
    FINAL_MODEL_PATH
)


print(
    "Final model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ==========================================================
# COMPARE WITH CURRENT MODELS
# ==========================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(
    "\nCNN Version 2 Test Accuracy:"
)

print(
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nPrevious experimental CNN:"
)

print(
    "cnn_model_new.keras = 43.06%"
)

print(
    "\nPrevious application CNN:"
)

print(
    "cnn_model.keras = 27.31%"
)


if test_accuracy >= 0.50:

    print(
        "\nExcellent: Version 2 reached "
        "50%+ test accuracy."
    )

elif test_accuracy > 0.4306:

    print(
        "\nImprovement: Version 2 is better "
        "than the previous 43.06% model."
    )

else:

    print(
        "\nVersion 2 did not improve over "
        "the previous 43.06% model."
    )


# ==========================================================
# SAVED FILES
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nSaved Files")
print("-" * 50)

print(
    "models/cnn_model_v2.keras"
)

print(
    "models/final_cnn_model_v2.keras"
)

print(
    "results/history_v2.csv"
)

print(
    "results/training_log_v2.csv"
)

print(
    "results/classification_report_v2.txt"
)

print(
    "results/confusion_matrix_v2.png"
)

print(
    "results/confusion_matrix_v2.csv"
)

print("\n" + "=" * 70)

