import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load model and extract model name
model_path = "TCN/tcn_ota_model.keras"
model = load_model(model_path)
model_name = "TCN"

# Load data
X_test = np.load("TCN/X_test.npy")
y_test = np.load("TCN/y_test.npy")

seq_len, num_features = X_test.shape[1], X_test.shape[2]
num_updates = 100

def compute_avg_accuracy(label_value):
    """Compute average accuracy per packet for given class label"""
    indices = np.where(y_test == label_value)[0]
    selected_indices = np.random.choice(indices, size=num_updates, replace=False)
    all_accs = np.zeros((num_updates, seq_len))

    for i, idx in enumerate(tqdm(selected_indices, desc=f"Processing {'OTA' if label_value == 1 else 'Normal'} samples")):
        sample = X_test[idx]
        true_label = y_test[idx]
        accs = []
        for t in tqdm(range(1, seq_len + 1), desc=f"Sample {i+1} packets", leave=False):
            partial_seq = sample[:t]
            padding = np.zeros((seq_len - t, num_features))
            input_seq = np.vstack((partial_seq, padding))
            input_seq = np.expand_dims(input_seq, axis=0)

            prob = model.predict(input_seq, verbose=0)[0][0]
            acc = prob * 100 if true_label == 1 else (1 - prob) * 100
            accs.append(acc)

        all_accs[i] = accs

    avg_accs = []
    for packet_idx in tqdm(range(seq_len), desc=f"Calculating {'OTA' if label_value == 1 else 'Normal'} averages"):
        packet_accs = all_accs[:, packet_idx]
        trimmed = np.delete(packet_accs, [packet_accs.argmax(), packet_accs.argmin()])
        avg = trimmed.mean()
        avg_accs.append(avg)

    return avg_accs

def find_stabilization_point(avg_accs, delta=0.1, min_acc=90):
    """
    Find the earliest packet index after which:
    - Accuracy stays >= min_acc
    - Changes between packets remain < delta
    - For the REST of the sequence (not just a short window)
    """
    avg_accs = np.array(avg_accs)
    seq_len = len(avg_accs)

    for start in range(seq_len):
        segment = avg_accs[start:]
        diffs = np.abs(np.diff(segment))

        if np.all(segment >= min_acc) and np.all(diffs < delta):
            return start + 1  # Convert to 1-based index
    return None




# Compute average accuracy curves
avg_accs_ota = compute_avg_accuracy(label_value=1)
avg_accs_normal = compute_avg_accuracy(label_value=0)

# Find stabilization points
stable_packet_ota = find_stabilization_point(avg_accs_ota, delta=0.1, min_acc=90)
stable_packet_normal = find_stabilization_point(avg_accs_normal, delta=0.1, min_acc=90)


# If not stable, default to last packet
if stable_packet_ota is None:
    stable_packet_ota = seq_len
    print("OTA prediction did not stabilize early; using last packet instead.")

if stable_packet_normal is None:
    stable_packet_normal = seq_len
    print("Normal prediction did not stabilize early; using last packet instead.")

print(f"OTA update prediction stabilizes at packet: {stable_packet_ota}")
print(f"Normal prediction stabilizes at packet: {stable_packet_normal}")


# Plot both on the same graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, seq_len + 1), avg_accs_ota, label='OTA Update Traffic', color='green')
plt.plot(range(1, seq_len + 1), avg_accs_normal, label='Normal Traffic', color='blue')

# Mark stabilization points on plot with annotation
if stable_packet_ota:
    plt.axvline(x=stable_packet_ota, color='green', linestyle=':', label='OTA Update Stabilization')
    plt.text(stable_packet_ota + 1, 
             avg_accs_ota[stable_packet_ota - 1],  # index offset since packets start at 1
             f'Packet {stable_packet_ota}', 
             color='green', fontsize=9, verticalalignment='bottom')

if stable_packet_normal:
    plt.axvline(x=stable_packet_normal, color='blue', linestyle=':', label='Normal Stabilization')
    plt.text(stable_packet_normal + 1, 
             avg_accs_normal[stable_packet_normal - 1], 
             f'Packet {stable_packet_normal}', 
             color='blue', fontsize=9, verticalalignment='bottom')

plt.title(f"Average Prediction Accuracy vs. Packets (100 Random Samples)\nModel: {model_name}")
plt.xlabel("Number of Packets")
plt.ylabel("Prediction Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save combined plot
filename = f"avg_accuracy_trimmed_100_random_combined_{model_name}.png"
plt.savefig(filename)
plt.show()
print(f"Saved plot as: {filename}")
