import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

# Input variable for gradient ascent
x = tf.Variable(tf.random.normal((1, 10), mean=0.0, stddev=1.0), dtype=tf.float32)

# Manual Gradient Ascent to Maximize Prediction
lr = 0.5
print("Running manual gradient ascent to maximize model output...")
for step in range(200):
    with tf.GradientTape() as tape:
        pred = model(x)
    
    grads = tape.gradient(pred, [x])[0]
    # Normalize grads to avoid vanishing gradients
    grad_norm = tf.norm(grads)
    if grad_norm > 1e-6:
        grads = grads / grad_norm
        
    x.assign_add(lr * grads)
    
    if step % 20 == 0:
        print(f"Step {step:03d} | Prediction: {pred.numpy()[0][0]:.6f}")

print("\nOptimal Input vector x that maximizes prediction:")
optimal_x = x.numpy()[0]
for idx, val in enumerate(optimal_x):
    print(f"Feature {idx:2d}: {val:10.6f}")

print("\nRunning manual gradient descent to minimize model output...")
x_min = tf.Variable(tf.random.normal((1, 10), mean=0.0, stddev=1.0), dtype=tf.float32)
for step in range(200):
    with tf.GradientTape() as tape:
        pred = model(x_min)
    
    grads = tape.gradient(pred, [x_min])[0]
    grad_norm = tf.norm(grads)
    if grad_norm > 1e-6:
        grads = grads / grad_norm
        
    x_min.assign_sub(lr * grads)
    
    if step % 20 == 0:
        print(f"Step {step:03d} | Prediction: {pred.numpy()[0][0]:.6f}")

print("\nOptimal Input vector x that minimizes prediction:")
optimal_x_min = x_min.numpy()[0]
for idx, val in enumerate(optimal_x_min):
    print(f"Feature {idx:2d}: {val:10.6f}")
