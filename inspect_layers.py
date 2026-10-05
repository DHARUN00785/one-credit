import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

low_risk = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
high_risk = np.array([[1.0, 9.0, 14.0, 130.0, 10.0, 80.0, 40.0, 80.0, 20.0, 16.0]], dtype=np.float32)

def print_layer_outputs(name, data):
    print(f"\n--- Output trace for {name} ---")
    current = data
    print(f"Input: {current}")
    for layer in model.layers:
        current = layer(current).numpy()
        print(f"Layer {layer.name}: mean={np.mean(current):.6f}, std={np.std(current):.6f}, min={np.min(current):.6f}, max={np.max(current):.6f}")
        # print first few elements if small
        if current.shape[-1] <= 10:
            print(f"  Values: {current}")

print_layer_outputs("low_risk", low_risk)
print_layer_outputs("high_risk", high_risk)
