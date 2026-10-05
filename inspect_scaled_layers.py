import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

mean_all = np.array([0.462, 6.088, 4.396, 43.096, 1.340, 16.022, 0.369, 0.198, 0.636, 7.423], dtype=np.float32)
std_all = np.array([0.499, 1.604, 2.985, 19.671, 1.705, 8.128, 1.267, 0.931, 1.263, 1.938], dtype=np.float32)

low_risk_patient = np.array([[0, 5, 2, 30, 0, 10, 0, 0, 0, 4]], dtype=np.float32)
high_risk_patient = np.array([[1, 8, 10, 85, 4, 35, 8, 6, 5, 9]], dtype=np.float32)

low_scaled = (low_risk_patient - mean_all) / std_all
high_scaled = (high_risk_patient - mean_all) / std_all

def trace(name, data):
    print(f"\n--- Trace for scaled {name} ---")
    current = data
    print(f"Scaled Input: {current}")
    for layer in model.layers:
        current = layer(current).numpy()
        print(f"Layer {layer.name}: mean={np.mean(current):.6f}, min={np.min(current):.6f}, max={np.max(current):.6f}")
        # Let's print the raw values if output is dense_5
        if layer.name == 'dense_5':
            print(f"  Final Sigmoid Prediction: {current[0][0]:.6f}")

trace("low_risk", low_scaled)
trace("high_risk", high_scaled)
