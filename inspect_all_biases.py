import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

for layer in model.layers:
    if hasattr(layer, 'weights') and len(layer.weights) > 0:
        w, b = layer.get_weights()
        print(f"\nLayer {layer.name}:")
        print(f"  Weights shape: {w.shape}, mean={np.mean(w):.6f}, min={np.min(w):.6f}, max={np.max(w):.6f}")
        print(f"  Biases shape:  {b.shape}, mean={np.mean(b):.6f}, min={np.min(b):.6f}, max={np.max(b):.6f}")
        if layer.name == 'dense_5':
            print("  Exact weights:")
            print(w.flatten())
            print("  Exact bias:")
            print(b)
