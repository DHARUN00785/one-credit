import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

# Let's define the feature names and typical raw ranges
features = [
    ("Gender", 0, 1),
    ("Age", 0, 9),
    ("Time in Hospital", 1, 14),
    ("Num Lab Procs", 1, 130),
    ("Num Procs", 0, 10),
    ("Num Meds", 1, 80),
    ("Num Outpatient", 0, 40),
    ("Num Emergency", 0, 80),
    ("Num Inpatient", 0, 20),
    ("Num Diagnoses", 1, 16)
]

# We will test Scenario A (Raw), Scenario B (Standard Scaled), and Scenario D (MinMax Scaled)
mean_all = np.array([0.462, 6.088, 4.396, 43.096, 1.340, 16.022, 0.369, 0.198, 0.636, 7.423], dtype=np.float32)
std_all = np.array([0.499, 1.604, 2.985, 19.671, 1.705, 8.128, 1.267, 0.931, 1.263, 1.938], dtype=np.float32)

min_vals = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
max_vals = np.array([1.0, 9.0, 14.0, 132.0, 6.0, 81.0, 42.0, 76.0, 21.0, 16.0], dtype=np.float32)

# Baseline is the mean/median raw profile
baseline_raw = np.array([[0, 6, 4, 43, 1, 16, 0, 0, 0, 7]], dtype=np.float32)

def predict(p, scaling_type):
    if scaling_type == 'raw':
        return model.predict(p, verbose=0)[0][0]
    elif scaling_type == 'std':
        p_scaled = (p - mean_all) / std_all
        return model.predict(p_scaled, verbose=0)[0][0]
    elif scaling_type == 'minmax':
        p_scaled = (p - min_vals) / (max_vals - min_vals)
        return model.predict(p_scaled, verbose=0)[0][0]

for scaling in ['raw', 'std', 'minmax']:
    print(f"\n================ Sensitivity Analysis under {scaling.upper()} SCALING ================")
    print(f"{'Feature':<20} | {'At Min Val':<12} | {'At Max Val':<12} | {'Direction':<10}")
    print("-" * 65)
    for idx, (name, min_val, max_val) in enumerate(features):
        # Set to min
        p_min = baseline_raw.copy()
        p_min[0, idx] = min_val
        pred_min = predict(p_min, scaling)
        
        # Set to max
        p_max = baseline_raw.copy()
        p_max[0, idx] = max_val
        pred_max = predict(p_max, scaling)
        
        direction = "INCREASES" if pred_max > pred_min else "DECREASES"
        if abs(pred_max - pred_min) < 1e-6:
            direction = "NO EFFECT"
            
        print(f"{name:<20} | {pred_min:10.6f} | {pred_max:10.6f} | {direction:<10}")
