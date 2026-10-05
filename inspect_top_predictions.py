import pandas as pd
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('readmission_model (1).keras')

df = pd.read_csv(r'c:\Users\HP\Downloads\diabetic_data.csv')

gender_map = {'Female': 0, 'Male': 1}
df['gender_val'] = df['gender'].map(gender_map).fillna(0).astype(int)

def encode_age(age_str):
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

feature_cols = [
    'gender_val', 'age_val', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses'
]

X_raw = df[feature_cols].values.astype(np.float32)

# Raw Predictions
preds_raw = model.predict(X_raw, batch_size=1024, verbose=0).flatten()
top_raw_indices = np.argsort(preds_raw)[::-1][:5]

print("\n--- TOP 5 PATIENTS BY RAW PREDICTION ---")
for idx in top_raw_indices:
    print(f"Index: {idx} | Prediction: {preds_raw[idx]:.6f}")
    for col_name, val in zip(feature_cols, X_raw[idx]):
        print(f"  {col_name:<20}: {val}")
    print()

# StandardScaler Predictions
mean_vals = np.mean(X_raw, axis=0)
std_vals = np.std(X_raw, axis=0)
X_std = (X_raw - mean_vals) / std_vals
preds_std = model.predict(X_std, batch_size=1024, verbose=0).flatten()
top_std_indices = np.argsort(preds_std)[::-1][:5]

print("\n--- TOP 5 PATIENTS BY STANDARD SCALED PREDICTION ---")
for idx in top_std_indices:
    print(f"Index: {idx} | Prediction: {preds_std[idx]:.6f}")
    for col_name, val in zip(feature_cols, X_raw[idx]):
        print(f"  {col_name:<20}: {val}")
    print()
