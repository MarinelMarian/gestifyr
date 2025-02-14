import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import os
import random

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
    for i in range(len(data)):
        seq = data.iloc[i].values
        if len(seq) != seq_length:
            i = i
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
        out = self.fc(out[-1])  # Take last output
        return out.squeeze()

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

def epoch_learn():
    model.train

    # Load dataset
    for file_path in file_paths:    
        X, y, gesture = get_x_y_gesture(file_path)

        dataset = GestureDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=sequence_length)

        # Loss and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.mean())
            loss.backward()
            optimizer.step()

def predict(epoch):
    # Prediction test using a random file from the dataset
    if epoch % 10 == 0:
        file_path = 'gesture_5__1739288434.csv'
    else:
        file_path = random.choice(file_paths)

    sample_input, sample_target, gesture = get_x_y_gesture(file_path)

    model.eval()
    with torch.no_grad():
        predictions = model(sample_input)

    diff = abs(gesture - predictions)
    if diff < 0.05:
        print(f"Epoch {epoch+1}/{epochs}, Predicted Gesture: {predictions}, Actual Gesture: {gesture}, Diff: {diff}, OK")
    else:
        print(f"Epoch {epoch+1}/{epochs}, Predicted Gesture: {predictions}, Actual Gesture: {gesture}, Diff: {diff}, !!!!!!!!")



# Model parameters
input_size = sequence_length
hidden_size = 52
num_layers = 2
model = RNNModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()

if os.path.exists('learn_gestures_3.pth'):
    print("Loading trained model")
    model_state = torch.load('learn_gestures_3.pth')
    model.load_state_dict(model_state)

# Training loop
epochs = 1000
for epoch in range(epochs):
    epoch_learn()
    predict(epoch)
    
print("Process complete.")

print("Saving trained model")
torch.save(model.state_dict(), 'learn_gestures_3.pth')
