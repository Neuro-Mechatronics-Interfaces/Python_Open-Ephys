import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utilities import rhd_utilities as rhd_utils
from utilities import emg_processing as emg_proc
from utilities.models import EMGCNN


# ==============================
# Load EMG Data
# ==============================
def load_emg_data(config_path):
    cfg = emg_proc.read_config_file(config_path)
    file_paths = rhd_utils.get_rhd_file_paths(cfg['root_directory'], verbose=True)
    metrics_filepath = os.path.join(cfg['root_directory'], cfg['metrics_filename'])
    metrics_data = pd.read_csv(metrics_filepath)
    print(f"Loaded metrics data from {metrics_filepath}: unique labels {metrics_data['Gesture'].unique()}")

    # Define mapping of gestures to numerical labels
    gesture_map = {
        "close": 0,
        "index": 1,
        "middle": 2,
        "open": 3,
        "pinky": 4,
        "rest": 5,
        "ring": 6,
        "thumb": 7
    }


    X, y = np.array([]), np.array([])
    for i, file in enumerate(file_paths):
        result, data_present = rhd_utils.load_file(file, verbose=False)
        if not data_present:
            continue

        emg_data = result['amplifier_data']  # (n_channels, n_samples)
        sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])

        # Apply preprocessing
        filtered_data = emg_proc.notch_filter(emg_data, fs=sample_rate, f0=60)
        filtered_data = emg_proc.butter_bandpass_filter(filtered_data, lowcut=20, highcut=400, fs=sample_rate, order=2, axis=1)
        rms_window_size = int(0.1 * sample_rate)
        rms_features = emg_proc.calculate_rms(filtered_data, rms_window_size)

        print(f"Processed file: {file}, RMS feature shape: {rms_features.shape}")

        X = np.concatenate((X, rms_features), axis=1) if X.size else rms_features

        # Make sure y includes the correct labels
        gesture = metrics_data[metrics_data['File Name'] == os.path.basename(file)]['Gesture'].values[0]
        y = np.concatenate((y, np.full(rms_features.shape[1], gesture_map[gesture]))) if y.size else np.full(rms_features.shape[1], gesture_map[gesture])

        #y = np.concatenate((y, np.full(rms_features.shape[1], i))) if y.size else np.full(rms_features.shape[1], i)


    X_tensor = torch.tensor(X.T, dtype=torch.float32) # (N_samples, N_channels)
    #X_tensor = X_tensor.unsqueeze(2)  # Adds a third dimension (sequence length = 1)

    # ✅ Fix: Ensure input has correct Conv1D shape (batch_size, channels, sequence_length)
    X_tensor = X_tensor.unsqueeze(-1)  # Shape becomes (N_samples, N_channels, 1)

    y_tensor = torch.tensor(y, dtype=torch.long)

    # Debugging: Print unique labels
    print(f"Unique labels in y_tensor: {torch.unique(y_tensor)}")

    return X_tensor, y_tensor


# ==============================
# Train & Evaluate Model
# ==============================
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    config_path = "/mnt/c/Users/NML/Desktop/hdemg_test/Jonathan/2024_11_11/CONFIG.txt"

    cfg = emg_proc.read_config_file(config_path)
    X_tensor, y_tensor = load_emg_data(config_path)

    # Dynamically adjust num_classes
    num_classes = len(torch.unique(y_tensor))
    print(f"Detected {num_classes} unique classes.")

    # Shape outputs should have the data as (N_channels, N_samples), and y as (N_samples,)
    print(f"X tensor shape: {X_tensor.shape}, y tensor shape: {y_tensor.shape}")

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = EMGCNN(num_classes=8, input_channels=128)
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=400)

    # Save trained model
    torch.save(model.state_dict(), os.path.join(cfg['root_directory'], "emg_cnn_model.pth"))

    # Plot Training Results
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.show()
