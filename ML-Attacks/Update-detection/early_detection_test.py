import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model and data
model = load_model("LSTM/lstm_ota_model.keras")
X_test = np.load("LSTM/X_test.npy")
y_test = np.load("LSTM/y_test.npy")

# Select one sample: first update
ota_index = np.where(y_test == 1)
normal_index = np.where(y_test == 0)

#sample = X_test[ota_index[0][10]] 
#true_label = y_test[ota_index[0][10]]

sample = X_test[normal_index[0][0]] 
true_label = y_test[normal_index[0][0]]

seq_len, num_features = sample.shape

predictions = []
accs = []

for t in range(1, seq_len + 1):
    partial_seq = sample[:t]
    padding = np.zeros((seq_len - t, num_features))  # pad with zeros
    input_seq = np.vstack((partial_seq, padding))
    input_seq = np.expand_dims(input_seq, axis=0)

    prob = model.predict(input_seq, verbose=0)[0][0]
    pred = int(prob > 0.5)

    predictions.append(pred)

    if true_label == 1:
        acc = prob * 100
    else:
        acc = (1 - prob) * 100
    
    accs.append(acc)

    print(f"Packet: {t} | Prediction: {pred} | Confidence: {acc:.4f} | True Label: {true_label}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, seq_len + 1), accs, label='Accuracy', color='blue')
plt.axhline(y=95, color='red', linestyle='--', label='Decision Threshold (95%)')
plt.title("Prediction Accruacy vs. Number of Packets (Single Normal Traffic Sample)")
plt.xlabel("Number of Packets")
plt.ylabel("Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_accuracy_single_sample_normal.png")
plt.show()
