# Speech Emotion Recognition Research Framework

## Overview

This repository contains the implementation of a **Speech Emotion Recognition (SER)** research framework developed as part of a thesis project.

The framework investigates multiple approaches for recognizing human emotions from speech, including:

* Conventional Machine Learning
* Convolutional Neural Networks (CNN)
* Attention-based models
* Time-series deep learning models
* Few-shot learning
* Meta-learning
* Cross-lingual evaluation
* Digital Signal Processing (DSP) analysis
* Attention visualization

The overall objective is to investigate how different learning paradigms and feature representations contribute to robust speech emotion recognition across different datasets and languages.

---

## Research Objective

Speech Emotion Recognition aims to automatically identify the emotional state expressed in a speech signal.

The research investigates the following questions:

1. How effectively can conventional machine learning models recognize emotions from speech?
2. How does deep learning compare with traditional machine learning approaches?
3. Can attention mechanisms improve emotion recognition?
4. How effective are time-series models for speech emotion recognition?
5. Can few-shot learning recognize emotions with very limited training examples?
6. How well do models generalize across different speech emotion datasets?
7. How does a model trained on English emotional speech perform on another language?
8. How do different digital signal processing techniques affect SER performance?

---

# System Architecture

The general processing pipeline used in the research is:

```text
                    Speech Audio
                         │
                         ▼
                Audio Preprocessing
                         │
                         ▼
                Feature Extraction
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       MFCC Features          Log-Mel Spectrogram
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Learning Models
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 Traditional ML      Deep Learning    Few-Shot Learning
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                 Emotion Prediction
                         │
                         ▼
              Performance Evaluation
```

---

# Datasets

The research uses multiple speech emotion datasets for experimentation and evaluation.

### RAVDESS

The **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** is used as one of the primary datasets.

The experiments use eight emotion categories:

| Emotion  |
| -------- |
| Neutral  |
| Calm     |
| Happy    |
| Sad      |
| Angry    |
| Fear     |
| Disgust  |
| Surprise |

The dataset contains speech recordings from multiple actors representing different emotional states.

### CREMA-D

The **Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)** is used for additional evaluation and cross-dataset experiments.

### TESS

The **Toronto Emotional Speech Set (TESS)** is also incorporated into the experimental evaluation.

### Dataset Combination

Experiments are performed using individual datasets as well as combinations of datasets to investigate model generalization.

---

# Feature Extraction

The framework investigates different representations of speech signals.

## Log-Mel Spectrogram

The main feature representation used in the deep learning pipeline is the **Log-Mel Spectrogram**.

The audio processing pipeline can be summarized as:

```text
Audio Signal
     │
     ▼
Short-Time Fourier Transform
     │
     ▼
Mel Filter Bank
     │
     ▼
Mel Spectrogram
     │
     ▼
Log Transformation
     │
     ▼
Log-Mel Spectrogram
```

The resulting representation can be used as a two-dimensional input for CNN-based architectures.

## MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** are also investigated as a speech representation, particularly in experiments involving sequential and attention-based models.

---

# Data Augmentation

Training data augmentation is used to increase the diversity of the training samples.

The implemented augmentation techniques include:

* Noise addition
* Pitch shifting
* Time stretching

Augmentation is applied to the training data while validation and testing data remain unmodified for evaluation.

---

# Models

The framework evaluates several categories of models.

## Conventional Machine Learning

The following conventional machine learning algorithms are investigated:

* Support Vector Machine (SVM)
* Random Forest (RF)
* K-Nearest Neighbors (KNN)
* Multi-Layer Perceptron (MLP)

These models provide baseline performance for comparison with deep learning approaches.

---

## Convolutional Neural Network

A CNN architecture is used to learn spatial patterns from Log-Mel Spectrogram representations.

The general architecture follows:

```text
Log-Mel Spectrogram
        │
        ▼
Convolution Layers
        │
        ▼
Pooling
        │
        ▼
Feature Representation
        │
        ▼
Fully Connected Layers
        │
        ▼
Emotion Classification
```

The trained CNN models are stored in the `models/` directory.

---

## Attention-Based Models

Attention mechanisms are investigated to determine whether the model can focus on emotionally informative regions of the speech representation.

The research includes experiments involving:

* Attention-LSTM
* Tiny Transformer
* Attention CNN

Attention visualization is also used to provide an interpretable representation of the regions receiving greater attention.

---

# Time-Series Models

Several modern time-series architectures are investigated for modeling temporal dependencies in speech features.

The evaluated architectures include:

* Reformer
* iTransformer
* TimesNet
* TimeXer

These models are evaluated on individual datasets and combined datasets to investigate their ability to model temporal patterns associated with emotional speech.

---

# Few-Shot Learning

The research also investigates **few-shot learning**, where the model is required to recognize emotion categories using only a small number of examples.

A **Prototypical Network (ProtoNet)** approach is used.

The basic process is:

```text
Support Set
    │
    ▼
Embedding Network
    │
    ▼
Class Prototypes
    │
    ▼
Query Samples
    │
    ▼
Distance to Prototypes
    │
    ▼
Predicted Emotion
```

Experiments include few-shot settings such as:

* 5-shot learning
* 10-shot learning

This investigates whether emotion recognition can be achieved when only a limited number of labeled examples are available.

---

# Cross-Lingual Evaluation

Cross-lingual experiments investigate the ability of speech emotion recognition models to generalize from one language to another.

The research includes:

```text
English Training Data
        │
        ▼
   Trained Model
        │
        ├──────────────► English Evaluation
        │
        ├──────────────► CREMA-D Evaluation
        │
        └──────────────► Urdu Evaluation
```

Additional experiments investigate domain adaptation using:

* Reptile meta-learning
* Domain-Adversarial Neural Network (DANN)

The objective is to study the effect of language and domain differences on emotion recognition performance.

---

# DSP Filter Analysis

Digital signal processing techniques are also investigated to determine their influence on SER performance.

Different filtering configurations are compared against an unfiltered signal.

The analysis investigates whether preprocessing through filtering improves or decreases emotion recognition performance.

The experiments indicate that **the unfiltered configuration performed best overall** in the evaluated setup.

---

# Experimental Setup

The experimental methodology follows a stratified dataset split.

```text
Dataset
   │
   ├── 70% Training
   │
   ├── 15% Validation
   │
   └── 15% Testing
```

Training augmentation is applied only to the training data.

The general experimental pipeline is:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Feature Extraction
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Model Training
   │
   ▼
Validation
   │
   ▼
Testing
   │
   ▼
Performance Evaluation
```

---

# Results Summary

The research evaluates multiple approaches across different datasets and experimental settings.

Some representative results obtained during the experiments include:

### Conventional Machine Learning — RAVDESS

| Model         | Accuracy |
| ------------- | -------: |
| SVM           |      44% |
| Random Forest |      62% |
| KNN           |      45% |
| MLP           |      51% |

### Conventional Machine Learning — TESS

| Model         | Accuracy |
| ------------- | -------: |
| SVM           |      97% |
| Random Forest |      99% |
| KNN           |      99% |
| MLP           |      99% |

### Time-Series Models

| Model        | CREMA-D | RAVDESS |
| ------------ | ------: | ------: |
| Reformer     |     43% |     72% |
| iTransformer |     50% |     43% |
| TimesNet     |     30% |     56% |

On the merged datasets, the experiments achieved:

| Model        | Accuracy |
| ------------ | -------: |
| TimeXer      |      88% |
| iTransformer |      74% |
| TimesNet     |      88% |

### Few-Shot Learning

A Prototypical Network experiment achieved approximately **70.92% accuracy** under the evaluated five-shot setting.

### Overall Performance

Representative results from the experimental study include:

| Approach               | Accuracy |
| ---------------------- | -------: |
| 2D-CNN                 |      92% |
| Baseline Supervised    |      87% |
| 1D-CNN                 |      84% |
| CNN-LSTM + Autoencoder |      75% |
| Attention-LSTM         |      75% |
| ProtoNet (5-shot)      |      70% |

These values represent results from the specific experimental configurations used in the research and should not be interpreted as universal performance benchmarks.

---

# Project Structure

```text
SpeechEmotionResearchDemo/
│
├── App.py
├── requirements.txt
├── runtime.txt
│
├── models/
│   ├── cnn_model.keras
│   └── final_cnn_model.keras
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Live_Prediction.py
│   ├── 3_Feature_Extraction.py
│   ├── 4_Attention.py
│   ├── 5_Model_Comparison.py
│   ├── 6_Cross_Lingual.py
│   └── 7_About_Research.py
│
├── saved_features/
│   └── label_encoder.pkl
│
├── utils/
│   ├── attention.py
│   ├── audio_features.py
│   ├── augmentation.py
│   ├── feature_utils.py
│   ├── prediction.py
│   └── visualizations.py
│
└── results/
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── history.csv
    ├── training_log.csv
    ├── train_split.csv
    ├── val_split.csv
    └── test_split.csv
```

---

# Directory Description

| Directory/File     | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| `models/`          | Trained neural network models                                                       |
| `pages/`           | Different research and demonstration modules                                        |
| `utils/`           | Feature extraction, prediction, visualization, augmentation and attention utilities |
| `saved_features/`  | Required serialized feature/label information                                       |
| `results/`         | Experimental results, evaluation reports and plots                                  |
| `requirements.txt` | Python dependencies                                                                 |
| `App.py`           | Main application entry point                                                        |

The original training dataset and large intermediate feature files are intentionally excluded from the repository to keep the repository lightweight.

---

# Technologies Used

The implementation uses the following technologies and libraries:

* Python
* TensorFlow / Keras
* PyTorch
* NumPy
* Pandas
* Librosa
* Scikit-learn
* OpenCV
* Matplotlib
* SoundFile
* Joblib

---

# Key Research Contributions

The framework brings together multiple perspectives for speech emotion recognition:

1. **Hybrid evaluation of traditional and deep learning approaches**
2. **Attention-based emotion recognition**
3. **Time-series modeling for speech representations**
4. **Few-shot emotion classification**
5. **Cross-dataset evaluation**
6. **Cross-lingual English-to-Urdu evaluation**
7. **DSP preprocessing and filtering analysis**
8. **Attention-based interpretability**
9. **Comparative evaluation across multiple speech emotion datasets**

---

# Reproducibility

The repository contains the source code, trained models, utility modules, configuration files, and selected experimental results required to understand and reproduce the implemented research pipeline.

Large datasets and intermediate training features are not included in the repository because of their size.

To reproduce the complete experiments, the corresponding datasets need to be obtained separately and placed in the appropriate dataset directories.

---

# Research Scope

This repository represents the implementation associated with a research study on **Speech Emotion Recognition using hybrid deep learning, attention mechanisms, time-series models, few-shot learning, and cross-lingual evaluation**.

The purpose of the repository is to provide a structured implementation of the experimental framework and supporting results for research analysis, evaluation, and academic demonstration.
