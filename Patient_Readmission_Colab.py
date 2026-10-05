# ============================================================================
# 🏥 Patient Hospital Readmission Prediction — Google Colab Notebook
# ============================================================================
# Deep Learning model to predict 30-day hospital readmission risk
# Dataset : UCI Diabetes 130-US Hospitals (1999–2008)
# Model   : Keras Sequential Neural Network
# ============================================================================
# HOW TO USE:
#   1. Open Google Colab → File → New Notebook
#   2. Copy each "# %%" section into a separate Colab cell
#   3. Run cells sequentially (Shift+Enter)
# ============================================================================


# %% [markdown]
# # 🏥 Patient Hospital Readmission Prediction
# **Objective:** Build a Deep Learning model to predict whether a diabetic
# patient will be readmitted to the hospital within 30 days.
#
# **Dataset:** [Diabetes 130-US Hospitals (UCI)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
#
# **Model Architecture:** Sequential Neural Network (Dense layers)


# %% — Cell 1: Install & Import Libraries
# ─────────────────────────────────────────────────────────────────────────────
!pip install -q kagglehub ucimlrepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score
)

warnings.filterwarnings('ignore')
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")


# %% — Cell 2: Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Option A — Download directly from UCI ML Repository
from ucimlrepo import fetch_ucirepo
diabetes_data = fetch_ucirepo(id=296)
df = diabetes_data.data.original.copy()

# Option B — If you uploaded diabetic_data.csv manually to Colab:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('diabetic_data.csv')

print(f"Dataset shape: {df.shape}")
df.head()


# %% — Cell 3: Initial Data Exploration
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
print(f"\nMissing values (top 10):")
missing = df.isnull().sum() + (df == '?').sum()
print(missing[missing > 0].sort_values(ascending=False).head(10))

print(f"\nReadmission distribution:")
print(df['readmitted'].value_counts())


# %% — Cell 4: Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

# 4a. Create binary target: 1 = readmitted within <30 days, 0 = otherwise
df['readmit_binary'] = (df['readmitted'] == '<30').astype(int)

print(f"Target distribution:")
print(df['readmit_binary'].value_counts())
print(f"\nReadmission rate: {df['readmit_binary'].mean()*100:.2f}%")

# 4b. Encode Gender
gender_map = {'Female': 0, 'Male': 1}
df['gender_val'] = df['gender'].map(gender_map).fillna(0).astype(int)

# 4c. Encode Age ranges → ordinal (0–9)
def encode_age(age_str):
    """Convert age range string to ordinal integer."""
    if not isinstance(age_str, str):
        return 0
    try:
        val = age_str.replace('[', '').replace(')', '').replace(']', '')
        if '-' in val:
            low = int(val.split('-')[0])
            return low // 10
        else:
            return int(val) // 10
    except:
        return 0

df['age_val'] = df['age'].apply(encode_age)

# 4d. Select the 10 features used in the model
feature_cols = [
    'gender_val',           # 0 = Female, 1 = Male
    'age_val',              # 0–9 (age bracket index)
    'time_in_hospital',     # 1–14 days
    'num_lab_procedures',   # 1–132
    'num_procedures',       # 0–6
    'num_medications',      # 1–81
    'number_outpatient',    # 0–42
    'number_emergency',     # 0–76
    'number_inpatient',     # 0–21
    'number_diagnoses'      # 1–16
]

X = df[feature_cols].values.astype(np.float32)
y = df['readmit_binary'].values.astype(np.float32)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")
print(f"Features: {feature_cols}")


# %% — Cell 5: Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('📊 Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 5a. Target distribution
ax = axes[0, 0]
colors = ['#2a9d8f', '#e63946']
df['readmit_binary'].value_counts().plot(kind='bar', ax=ax, color=colors)
ax.set_title('Readmission Distribution')
ax.set_xticklabels(['No (<30 days)', 'Yes (<30 days)'], rotation=0)
ax.set_ylabel('Count')

# 5b. Age distribution
ax = axes[0, 1]
df['age_val'].hist(bins=10, ax=ax, color='#0077cc', edgecolor='white', alpha=0.8)
ax.set_title('Age Distribution')
ax.set_xlabel('Age Group Index (0=0-10, 9=90-100)')

# 5c. Time in hospital
ax = axes[0, 2]
df['time_in_hospital'].hist(bins=14, ax=ax, color='#f4a261', edgecolor='white', alpha=0.8)
ax.set_title('Time in Hospital (Days)')

# 5d. Number of medications
ax = axes[1, 0]
df['num_medications'].hist(bins=30, ax=ax, color='#e76f51', edgecolor='white', alpha=0.8)
ax.set_title('Number of Medications')

# 5e. Number of diagnoses
ax = axes[1, 1]
df['number_diagnoses'].hist(bins=16, ax=ax, color='#264653', edgecolor='white', alpha=0.8)
ax.set_title('Number of Diagnoses')

# 5f. Correlation heatmap of features
ax = axes[1, 2]
feature_df = pd.DataFrame(X, columns=feature_cols)
feature_df['readmitted'] = y
corr = feature_df.corr()['readmitted'].drop('readmitted').sort_values()
corr.plot(kind='barh', ax=ax, color=['#e63946' if v > 0 else '#2a9d8f' for v in corr])
ax.set_title('Feature Correlation with Readmission')
ax.set_xlabel('Correlation Coefficient')

plt.tight_layout()
plt.show()


# %% — Cell 6: Feature Statistics
# ─────────────────────────────────────────────────────────────────────────────
stats_df = pd.DataFrame(X, columns=feature_cols)
print("Feature Statistics:")
print(stats_df.describe().round(3).to_string())

# Readmission rates by age group
print("\n\nReadmission Rate by Age Group:")
age_readmit = df.groupby('age_val')['readmit_binary'].mean() * 100
for age, rate in age_readmit.items():
    print(f"  Age [{age*10}-{age*10+10}): {rate:.2f}%")


# %% — Cell 7: Train-Test Split
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # Maintain class balance in splits
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"\nTraining readmission rate: {y_train.mean()*100:.2f}%")
print(f"Test readmission rate:     {y_test.mean()*100:.2f}%")


# %% — Cell 8: Build the Model
# ─────────────────────────────────────────────────────────────────────────────
# Architecture matches the saved readmission_model (1).keras:
#   Input(10) → Dense(32, relu) → Dense(48, relu) → Dense(1, sigmoid)

model = Sequential([
    Input(shape=(10,), name='input_layer_1'),
    Dense(32, activation='relu', name='dense_3'),
    Dense(48, activation='relu', name='dense_4'),
    Dense(1, activation='sigmoid', name='dense_5')
], name='sequential_1')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# %% — Cell 9: Train the Model
# ─────────────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]
)


# %% — Cell 10: Training History Visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('📈 Training History', fontsize=14, fontweight='bold')

# Loss curve
ax1.plot(history.history['loss'], label='Train Loss', color='#0077cc', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', color='#e63946', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#0077cc', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e63946', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% — Cell 11: Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
# Predict on test set
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Test loss and accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Classification Report
print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(
    y_test, y_pred,
    target_names=['Not Readmitted', 'Readmitted (<30d)']
))


# %% — Cell 12: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Not Readmitted', 'Readmitted'],
    yticklabels=['Not Readmitted', 'Readmitted'],
    ax=ax, linewidths=0.5
)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('🔍 Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nTrue Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")


# %% — Cell 13: ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#0077cc', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.15, color='#0077cc')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('📉 ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AUC Score: {roc_auc:.4f}")


# %% — Cell 14: Prediction Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('📊 Prediction Distribution', fontsize=14, fontweight='bold')

# Overall distribution
ax1.hist(y_pred_prob, bins=50, color='#0077cc', edgecolor='white', alpha=0.8)
ax1.axvline(x=0.5, color='#e63946', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Count')
ax1.set_title('Overall Prediction Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# By class
ax2.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.6, color='#2a9d8f', label='Not Readmitted')
ax2.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.6, color='#e63946', label='Readmitted')
ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Count')
ax2.set_title('By Actual Class')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nPrediction Statistics:")
print(f"  Mean:   {np.mean(y_pred_prob):.6f}")
print(f"  Median: {np.median(y_pred_prob):.6f}")
print(f"  Min:    {np.min(y_pred_prob):.6f}")
print(f"  Max:    {np.max(y_pred_prob):.6f}")
print(f"  Predictions > 0.5: {np.sum(y_pred_prob > 0.5)} / {len(y_pred_prob)}")


# %% — Cell 15: Test with Sample Patients
# ─────────────────────────────────────────────────────────────────────────────
# Feature order: gender, age, time_in_hospital, num_lab_procedures,
#   num_procedures, num_medications, number_outpatient, number_emergency,
#   number_inpatient, number_diagnoses

sample_patients = {
    "Low-Risk Patient": np.array([[0, 5, 2, 30, 0, 10, 0, 0, 0, 4]], dtype=np.float32),
    "Medium-Risk Patient": np.array([[1, 7, 6, 60, 2, 25, 2, 1, 1, 8]], dtype=np.float32),
    "High-Risk Patient": np.array([[1, 8, 10, 85, 4, 35, 8, 6, 5, 9]], dtype=np.float32),
}

print("=" * 60)
print("SAMPLE PATIENT PREDICTIONS")
print("=" * 60)
for name, patient_data in sample_patients.items():
    pred = model.predict(patient_data, verbose=0)[0][0]
    risk = "🔴 HIGH" if pred >= 0.30 else "🟠 MODERATE" if pred >= 0.12 else "🟢 LOW"
    print(f"\n{name}:")
    print(f"  Features: {patient_data[0].tolist()}")
    print(f"  Probability: {pred:.6f} ({pred*100:.2f}%)")
    print(f"  Risk Level: {risk}")


# %% — Cell 16: Save Model
# ─────────────────────────────────────────────────────────────────────────────
# Save in Keras format
model.save('readmission_model.keras')
print("✅ Model saved as 'readmission_model.keras'")

# Download the model file (uncomment to use)
# from google.colab import files
# files.download('readmission_model.keras')


# %% — Cell 17: Feature Importance (Sensitivity Analysis)
# ─────────────────────────────────────────────────────────────────────────────
feature_names = [
    'Gender', 'Age', 'Time in Hospital', 'Lab Procedures',
    'Procedures', 'Medications', 'Outpatient Visits',
    'Emergency Visits', 'Inpatient Visits', 'Diagnoses'
]

# Baseline: mean values
baseline = np.mean(X_train, axis=0, keepdims=True)
baseline_pred = model.predict(baseline, verbose=0)[0][0]

importance = []
for i in range(10):
    perturbed = baseline.copy()
    perturbed[0, i] = np.max(X_train[:, i])  # Set to max observed value
    pred_max = model.predict(perturbed, verbose=0)[0][0]
    
    perturbed[0, i] = np.min(X_train[:, i])  # Set to min observed value
    pred_min = model.predict(perturbed, verbose=0)[0][0]
    
    importance.append(abs(pred_max - pred_min))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importance)
ax.barh(
    [feature_names[i] for i in sorted_idx],
    [importance[i] for i in sorted_idx],
    color='#0077cc', edgecolor='white', height=0.6
)
ax.set_xlabel('Prediction Sensitivity (|max - min|)')
ax.set_title('🧠 Feature Importance (Sensitivity Analysis)', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# %% — Cell 18: Model Summary & Final Report
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("📋 FINAL MODEL REPORT")
print("=" * 60)
print(f"""
Project:        Patient Hospital Readmission Prediction
Dataset:        UCI Diabetes 130-US Hospitals (1999–2008)
Total Samples:  {len(X)}
Features:       {len(feature_cols)} clinical features
Train/Test:     80/20 split (stratified)

Model Architecture:
  Input(10) → Dense(32, ReLU) → Dense(48, ReLU) → Dense(1, Sigmoid)

Compilation:
  Optimizer: Adam
  Loss:      Binary Crossentropy
  Metrics:   Accuracy

Results:
  Test Accuracy: {test_acc:.4f}
  AUC Score:     {roc_auc:.4f}
  Test Loss:     {test_loss:.4f}

Features Used:
  {chr(10).join(f'  {i+1}. {name}' for i, name in enumerate(feature_names))}

Risk Thresholds (for Streamlit app):
  Low Risk:      < 12%
  Moderate Risk: 12% – 30%
  High Risk:     ≥ 30%
""")
print("✅ Notebook execution complete!")
