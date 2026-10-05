import tensorflow as tf
import numpy as np

def inspect_weights(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model Layers:")
    for layer in model.layers:
        print(f"Layer: {layer.name}, Class: {layer.__class__.__name__}")
        weights = layer.get_weights()
        if weights:
            print(f"  Weights shape: {[w.shape for w in weights]}")
            
    # Inspect first dense layer weights
    first_dense = model.get_layer("dense_3")
    w, b = first_dense.get_weights()
    # w shape is (10, 32)
    # Let's see the contribution of each of the 10 features to the units in dense_3
    # We can also compute the gradients or look at the overall sensitivity.
    
    print("\nWeight statistics for the 10 input features:")
    feature_names = [
        "Gender", "Age", "Time in Hospital", "Num Lab Procs", "Num Procs",
        "Num Meds", "Num Outpatient", "Num Emergency", "Num Inpatient", "Num Diagnoses"
    ]
    for i, name in enumerate(feature_names):
        feature_weights = w[i]
        pos_sum = np.sum(feature_weights[feature_weights > 0])
        neg_sum = np.sum(feature_weights[feature_weights < 0])
        mean_val = np.mean(feature_weights)
        print(f"{i:2d}. {name:<20}: Mean weight={mean_val:6.3f}, Pos sum={pos_sum:6.3f}, Neg sum={neg_sum:6.3f}")

if __name__ == "__main__":
    inspect_weights("readmission_model (1).keras")
