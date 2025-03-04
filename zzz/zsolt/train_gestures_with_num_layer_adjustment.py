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

# load all the files in the file path
file_paths = []
# Get all files in the current directory
for file_name in os.listdir("./for-training/"):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("gesture_") and file_name.endswith(".csv"):
        file_paths.append(file_name)
file_paths.sort()

# load all the files in the test path
test_paths = [];
# Get all files in the current directory
for file_name in os.listdir("./for-testing/"):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("gesture_") and file_name.endswith(".csv"):
        test_paths.append(file_name)
test_paths.sort()

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
        label = gesture
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
        X, y, gesture = get_x_y_gesture('./for-training/' + file_path)

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
    stat = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        # 5: 0,
    }

    total_predictions = 0
    correct_predictions = 0
    prediction_difference = 0

    # Prediction test using test files
    for run_nr in range(5):
        for file_path in test_paths:
            total_predictions += 1

            file_path = random.choice(test_paths)

            sample_input, sample_target, gesture = get_x_y_gesture("./for-testing/" + file_path)
            gesture_key = int(gesture*10)

            model.eval()
            with torch.no_grad():
                prediction = model(sample_input)

            diff = abs(gesture - prediction)
            if diff < 0.05:
                stat[gesture_key] += 1
                correct_predictions += 1
                prediction_difference += diff

    correct_predictions_percent = (correct_predictions * 100) / total_predictions
    for key in stat:
        if stat[key] > 0:
            stat[key] = (stat[key] * 100) / correct_predictions

    formatted_stat = [f"{stat[key]:.2f}" for key in stat]
    print(f"Epoch {epoch+1}/{epochs} | total predictions: {total_predictions} | correct: {correct_predictions_percent}% | spread: {formatted_stat} | diff: {prediction_difference:.4f}")

    # if the total correct predictions are > X%
    # then check the individual achivements
    if correct_predictions_percent > 85:
        print(f"Prediction% is {correct_predictions_percent}")

        ideal_percentage = 100 / len(stat)
        for key in stat:
            # if one of the gestures has more than double the ideal percentage
            # that means that it has a bias over the other gestures in the model
            if stat[key] > ideal_percentage * 2:
                # we need to increase the num_layers and restart the training
                print(f"Gesture {key} has a precision of {stat[key]}% and seems to be a bias")
                return True



    return False


# Model parameters
input_size = sequence_length
hidden_size = input_size * 2
num_layers = 1
model = RNNModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()

# if a saved model exists, load it into our model
# if os.path.exists('learn_gestures_7.pth'):
#     print("Loading trained model")
#     model_state = torch.load('learn_gestures_7.pth')
#     model.load_state_dict(model_state)

# Training loop
epochs = 2000
while True:
    for epoch in range(epochs):
        epoch_learn()
        restart = predict(epoch)
        if restart:
            num_layers += 1
            model = RNNModel(input_size, hidden_size, num_layers)
            print(f"We are increasing num_layers to {num_layers} and resetting the model, training starts again")
            print(f"Restarting epochs with input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}")
            break

    if epoch == 2000:
        break
    
print("Process complete.")

print("Saving trained model")
torch.save(model.state_dict(), 'learn_gestures_7.pth')
