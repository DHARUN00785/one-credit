import pandas as pd
import numpy as np
import tensorflow as tf
# Load model
model = tf.keras.models.load_model('readmission_model (1).keras')

# Load dataset
df = pd.read_csv(r'c:\Users\HP\Downloads\diabetic_data.csv')

# Preprocess features as in app.py
# Gender: Female -> 0, Male -> 1, others -> 0
gender_map = {'Female': 0, 'Male': 1}
df['gender_val'] = df['gender'].map(gender_map).fillna(0).astype(int)

# Age: '[0-10)' -> 0, '[10-20)' -> 1, ..., '[90-100)' -> 9
def encode_age(age_str):
    if not isinstance(age_str, str):
        return 0
    # Expected formats: "[0-10)", "55", etc.
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

# Align with app.py features
feature_cols = [
    'gender_val', 'age_val', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses'
]

X_raw = df[feature_cols].values.astype(np.float32)

# Verify shapes
print(f"X_raw shape: {X_raw.shape}")

# Evaluate raw inputs
preds_raw = model.predict(X_raw, batch_size=1024, verbose=0).flatten()
print(f"\nRaw Predictions:")
print(f"  Mean prediction: {np.mean(preds_raw):.6f}")
print(f"  Min prediction:  {np.min(preds_raw):.6f}")
print(f"  Max prediction:  {np.max(preds_raw):.6f}")
print(f"  Predictions > 0.1: {np.sum(preds_raw > 0.1)} / {len(preds_raw)}")
print(f"  Predictions > 0.2: {np.sum(preds_raw > 0.2)} / {len(preds_raw)}")
print(f"  Predictions > 0.5: {np.sum(preds_raw > 0.5)} / {len(preds_raw)}")

# Evaluate StandardScaler inputs
mean_vals = np.mean(X_raw, axis=0)
std_vals = np.std(X_raw, axis=0)
X_std = (X_raw - mean_vals) / std_vals
preds_std = model.predict(X_std, batch_size=1024, verbose=0).flatten()
print(f"\nStandardScaler Predictions:")
print(f"  Mean prediction: {np.mean(preds_std):.6f}")
print(f"  Min prediction:  {np.min(preds_std):.6f}")
print(f"  Max prediction:  {np.max(preds_std):.6f}")
print(f"  Predictions > 0.1: {np.sum(preds_std > 0.1)} / {len(preds_std)}")
print(f"  Predictions > 0.2: {np.sum(preds_std > 0.2)} / {len(preds_std)}")
print(f"  Predictions > 0.5: {np.sum(preds_std > 0.5)} / {len(preds_std)}")

# Evaluate MinMaxScaler inputs
min_vals = np.min(X_raw, axis=0)
max_vals = np.max(X_raw, axis=0)
X_mm = (X_raw - min_vals) / (max_vals - min_vals)
preds_mm = model.predict(X_mm, batch_size=1024, verbose=0).flatten()
print(f"\nMinMaxScaler Predictions:")
print(f"  Mean prediction: {np.mean(preds_mm):.6f}")
print(f"  Min prediction:  {np.min(preds_mm):.6f}")
print(f"  Max prediction:  {np.max(preds_mm):.6f}")
print(f"  Predictions > 0.1: {np.sum(preds_mm > 0.1)} / {len(preds_mm)}")
print(f"  Predictions > 0.2: {np.sum(preds_mm > 0.2)} / {len(preds_mm)}")
print(f"  Predictions > 0.5: {np.sum(preds_mm > 0.5)} / {len(preds_mm)}")
