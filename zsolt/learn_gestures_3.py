import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import os

# Define datasets
file_paths = []
# Get all files in the current directory
for file_name in os.listdir("."):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("gesture_") and file_name.endswith(".csv"):
        file_paths.append(file_name)

file_paths.sort()

# Normalize data
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# Convert to sequences
sequence_length = 52
target = "Gesture"

def create_sequences(data, seq_length, gesture):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length].values
        label = gesture;
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)


# Dataset and DataLoader
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out.squeeze()

# Model parameters
input_size = sequence_length
hidden_size = 32
num_layers = 2
model = RNNModel(input_size, hidden_size, num_layers)

def get_x_y_gesture(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    df = df.apply(normalize)

    gesture = 0

    match = re.search(r'gesture_(\d+)_', file_path)
    if match:
        gesture = float(int(match.group(1))/10)
    else:
        print("Gesture not identified in file name {file_path}. Aborting")
        return

    X, y = create_sequences(df, sequence_length, gesture)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    return X, y, gesture

def epoch_learn(epoch):
    # Load dataset
    file_path = file_paths[epoch]
    
    X, y, gesture = get_x_y_gesture(file_path)

    dataset = GestureDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000000000000001)

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if "loss" in locals():
        print(f"Epoch {epoch+1}/{epochs}, File {file_path}, Gesture {gesture}, Loss: {loss.item():.4f}")
    else:
        print(f"Epoch {epoch+1}/{epochs}, File {file_path}, Gesture {gesture}, Loss: !!!UNDEFINED!!!")

# Training loop
epochs = len(file_paths)
for epoch in range(epochs):
    epoch_learn(epoch)
    

print("Training complete.")


# Prediction test using the 3rd row from the dataset
sample_input, sample_target, gesture = get_x_y_gesture(file_paths[2])
# sample_input = sample_input.unsqueeze(0)  # Reshape to match model input
model.eval()
with torch.no_grad():
    predictions = model(sample_input)
print(f"Predicted Gesture: {predictions}, Actual Gesture: {gesture}")
