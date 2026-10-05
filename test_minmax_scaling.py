import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

# Min and max values from the dataset
min_vals = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
max_vals = np.array([1.0, 9.0, 14.0, 132.0, 6.0, 81.0, 42.0, 76.0, 21.0, 16.0], dtype=np.float32)

low_risk_patient = np.array([[0, 5, 2, 30, 0, 10, 0, 0, 0, 4]], dtype=np.float32)
high_risk_patient = np.array([[1, 8, 10, 85, 4, 35, 8, 6, 5, 9]], dtype=np.float32)

def predict_minmax(p):
    p_scaled = (p - min_vals) / (max_vals - min_vals)
    return model.predict(p_scaled, verbose=0)[0][0]

print(f"MinMax - Low Risk Patient:  {predict_minmax(low_risk_patient):.6f}")
print(f"MinMax - High Risk Patient: {predict_minmax(high_risk_patient):.6f}")

print("\nMinMax Interpolation test (low risk -> high risk):")
print(f"{'Alpha':<6} | {'MinMax Prediction':<20}")
print("-" * 35)
for alpha in np.linspace(0, 1, 11):
    p = low_risk_patient + alpha * (high_risk_patient - low_risk_patient)
    pred = predict_minmax(p)
    print(f"{alpha:5.1f} | {pred:20.6f}")
