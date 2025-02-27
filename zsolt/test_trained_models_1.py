import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import numpy as np
import time

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

# Normalize data
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# create the pytorch friendly test or training sequence
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

# get the content of the file in a pytorch format
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




# Model parameters
sequence_length = 52
input_size = sequence_length
hidden_size = 52
num_layers = 2

# load all the files in the file path
model_file_paths = []
# Get all files in the current directory
for file_name in os.listdir("./trained_models/"):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("trained_model_OK") and file_name.endswith(".pth"):
        model_file_paths.append(file_name)
model_file_paths.sort(reverse = True)

# load all the files in the test path
test_paths = [];
# Get all files in the current directory
for file_name in os.listdir("./for-testing/"):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("gesture_") and file_name.endswith(".csv"):
        test_paths.append(file_name)
test_paths.sort()

# csv data to dump at the end
csv_content = 'Model name,Correct,Correct%,Diff,G1,G2,G3,G4,G5' + os.linesep
no_of_test_walkthrough = 40

# Parse through trained models
for model_file_path in model_file_paths:
    # (Re)Define the model
    model = RNNModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()

    # load the trained model
    model_state = torch.load('./trained_models/' + model_file_path)
    model.load_state_dict(model_state)
    model.eval()

    # initialize total variables
    total_predictions = 0
    total_correct_predictions = 0
    total_difference = 0.00

    # statistic
    stat = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        # 5: 0,
    }

    # Parse through test files 10 times
    for run_no in range(no_of_test_walkthrough):
        for test_file_path in test_paths:
            total_predictions += 1

            # get the sample
            sample_input, sample_target, gesture = get_x_y_gesture("./for-testing/" + test_file_path)

            # generate a prediction
            with torch.no_grad():
                prediction = model(sample_input)
            
            # calculate the difference and evaliate the prediction result
            diff = abs(gesture - prediction)
            if diff < 0.05:
                # prediction is good
                total_correct_predictions += 1
                total_difference += diff
                stat[int(gesture*10)] += 1

    for key in stat:
        stat[key] = int((100 * stat[key]) / total_correct_predictions)

    print(f"File: {model_file_path} | TPredictions: {total_predictions} | TCorrect: {total_correct_predictions} {((100 * total_correct_predictions) / total_predictions):.2f}% | Gestures: {stat} | TDiff: {total_difference:.4f}")
    csv_content += f"{model_file_path},{total_correct_predictions},{((100 * total_correct_predictions) / total_predictions):.2f},{total_difference:.4f}"
    for key in stat:
        csv_content += f",{stat[key]}"
    csv_content += os.linesep

np.savetxt('test_trained_models_result_w' + str(no_of_test_walkthrough) + '_' + str(int(time.time())) + '.csv', [csv_content], fmt='%s')
