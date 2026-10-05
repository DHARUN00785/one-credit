import tensorflow as tf
import sys

def inspect_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model Loaded Successfully")
        print("\nModel Summary:")
        model.summary()
        
        print("\nInput Configuration:")
        for i, input_tensor in enumerate(model.inputs):
            print(f"Input {i}: shape={input_tensor.shape}, dtype={input_tensor.dtype}, name={input_tensor.name}")
            
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)

if __name__ == "__main__":
    inspect_model("readmission_model.h5")
