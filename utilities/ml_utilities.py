import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ==============================
# Model building
# ==============================
class EMGCNN(nn.Module):
    def __init__(self, num_classes=7, input_channels=8):
        super(EMGCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(1, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(1, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        #self.fc = nn.Linear(64 * 5, num_classes)  # Adjust based on your final feature map size
        # ✅ Compute final feature map size dynamically (Adjust this based on pooling)
        self.fc = nn.Linear(64 * 1, num_classes)  # Adjusted for sequence_length = 1
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Saved trained model to {path}")



# ==============================
# Training & Evaluation
# ==============================
def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
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
# Transformations
# ==============================
def convert_lists_to_tensors(X_list, y_list):
    """Converts lists of EMG features and labels into PyTorch tensors."""
    X_tensor = torch.tensor(np.vstack(X_list), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(np.concatenate(y_list), dtype=torch.long)

    print(f"Final X tensor shape: {X_tensor.shape}, Final y tensor shape: {y_tensor.shape}")
    return X_tensor, y_tensor


# ==============================
# Plotting
# ==============================

def plot_training_metrics(train_losses, val_losses, val_accuracies, save_fig=False):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    if save_fig:
        plt.savefig("loss_plot.png")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    if save_fig:
        plt.savefig("accuracy_plot.png")
    plt.show()
