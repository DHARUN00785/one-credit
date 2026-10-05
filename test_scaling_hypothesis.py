import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

# Define a low risk patient (raw values)
# Gender: Female (0)
# Age: [50-60) (5)
# Time in Hospital: 2 days
# Num Lab Procs: 30
# Num Procs: 0
# Num Meds: 10
# Num Outpatient: 0
# Num Emergency: 0
# Num Inpatient: 0
# Num Diagnoses: 4
low_risk_patient = np.array([[0, 5, 2, 30, 0, 10, 0, 0, 0, 4]], dtype=np.float32)

# Define a high risk patient (raw values)
# Gender: Male (1)
# Age: [80-90) (8)
# Time in Hospital: 10 days
# Num Lab Procs: 85
# Num Procs: 4
# Num Meds: 35
# Num Outpatient: 8
# Num Emergency: 6
# Num Inpatient: 5
# Num Diagnoses: 9
high_risk_patient = np.array([[1, 8, 10, 85, 4, 35, 8, 6, 5, 9]], dtype=np.float32)

# UCI stats
mean_all = np.array([0.462, 6.088, 4.396, 43.096, 1.340, 16.022, 0.369, 0.198, 0.636, 7.423], dtype=np.float32)
std_all = np.array([0.499, 1.604, 2.985, 19.671, 1.705, 8.128, 1.267, 0.931, 1.263, 1.938], dtype=np.float32)

# Scenarios to test
def predict_scenario_raw(p):
    return model.predict(p, verbose=0)[0][0]

def predict_scenario_scale_all(p):
    p_scaled = (p - mean_all) / std_all
    return model.predict(p_scaled, verbose=0)[0][0]

def predict_scenario_scale_num_only(p):
    # Scale only features from index 2 to 9
    p_scaled = p.copy()
    p_scaled[0, 2:] = (p[0, 2:] - mean_all[2:]) / std_all[2:]
    return model.predict(p_scaled, verbose=0)[0][0]

print("--- Low Risk Patient ---")
print(f"Scenario A (Raw):       {predict_scenario_raw(low_risk_patient):.6f}")
print(f"Scenario B (Scale All): {predict_scenario_scale_all(low_risk_patient):.6f}")
print(f"Scenario C (Scale Num): {predict_scenario_scale_num_only(low_risk_patient):.6f}")

print("\n--- High Risk Patient ---")
print(f"Scenario A (Raw):       {predict_scenario_raw(high_risk_patient):.6f}")
print(f"Scenario B (Scale All): {predict_scenario_scale_all(high_risk_patient):.6f}")
print(f"Scenario C (Scale Num): {predict_scenario_scale_num_only(high_risk_patient):.6f}")

# Interpolation test (from low risk to high risk)
print("\n--- Interpolation test (low risk -> high risk) ---")
print(f"{'Alpha':<6} | {'Scenario A (Raw)':<18} | {'Scenario B (Scale All)':<22} | {'Scenario C (Scale Num)':<22}")
print("-" * 75)
for alpha in np.linspace(0, 1, 11):
    p = low_risk_patient + alpha * (high_risk_patient - low_risk_patient)
    pred_raw = predict_scenario_raw(p)
    pred_all = predict_scenario_scale_all(p)
    pred_num = predict_scenario_scale_num_only(p)
    print(f"{alpha:5.1f} | {pred_raw:18.6f} | {pred_all:22.6f} | {pred_num:22.6f}")
