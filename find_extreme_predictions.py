import tensorflow as tf
# pyrefly: ignore [missing-import]
import numpy as np

model = tf.keras.models.load_model('readmission_model (1).keras')

# Let's run a random search over reasonable input ranges
np.random.seed(42)
num_samples = 100000

# Feature ranges from app.py:
# Gender: 0 or 1
# Age: 0 to 9
# Time in Hospital: 1 to 14
# Num Lab Procs: 1 to 130
# Num Procs: 0 to 10
# Num Meds: 1 to 80
# Num Outpatient: 0 to 40
# Num Emergency: 0 to 80
# Num Inpatient: 0 to 20
# Num Diagnoses: 1 to 16

genders = np.random.randint(0, 2, num_samples)
ages = np.random.randint(0, 10, num_samples)
time_hosp = np.random.randint(1, 15, num_samples)
num_lab = np.random.randint(1, 131, num_samples)
num_proc = np.random.randint(0, 11, num_samples)
num_meds = np.random.randint(1, 81, num_samples)
num_out = np.random.randint(0, 41, num_samples)
num_emerg = np.random.randint(0, 81, num_samples)
num_inpat = np.random.randint(0, 21, num_samples)
num_diag = np.random.randint(1, 17, num_samples)

X = np.stack([genders, ages, time_hosp, num_lab, num_proc, num_meds, num_out, num_emerg, num_inpat, num_diag], axis=1).astype(np.float32)

preds = model.predict(X, batch_size=10000)

max_idx = np.argmax(preds)
min_idx = np.argmin(preds)

print(f"Max prediction: {preds[max_idx][0]:.6f}")
print("Input for max prediction:")
features = [
    "Gender", "Age", "Time in Hospital", "Num Lab Procs", "Num Procs",
    "Num Meds", "Num Outpatient", "Num Emergency", "Num Inpatient", "Num Diagnoses"
]
for name, val in zip(features, X[max_idx]):
    print(f"  {name}: {val}")

print(f"\nMin prediction: {preds[min_idx][0]:.6f}")
print("Input for min prediction:")
for name, val in zip(features, X[min_idx]):
    print(f"  {name}: {val}")

# Let's count how many samples have prediction > 0.5
above_threshold = np.sum(preds > 0.5)
print(f"\nNumber of samples with prediction > 0.5: {above_threshold} out of {num_samples} ({above_threshold/num_samples*100:.3f}%)")
print(f"Mean prediction: {np.mean(preds):.6f}")
