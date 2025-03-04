#import for prediction function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#import for testing
import pandas as pd
import numpy as np

## PREDICTION
trained_model_file_path = "./trained_models/trained_model_OK15_e1484_p93__1740582936.pth"

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

# gesture prediction by a sample input
# default input_size, hidden_size and num_layers are the values this model was trained with
# return value is a number from 1 to 5, referring to the gesture number that is predicts
def predict_gesture_by_sample_input(sample_input, input_size = 52, hidden_size = 52, num_layers = 2) -> int:
    # Define the model
    model = RNNModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()

    # load the trained model
    model_state = torch.load(trained_model_file_path)
    model.load_state_dict(model_state)
    model.eval()

    # generate a prediction
    with torch.no_grad():
        prediction = model(sample_input)

    return int(prediction * 10)

## TESTING 
sequence_length = 52

# Normalize data
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

# define the function to get sample_input from file
def get_sample_input_from_file(test_file_path):
    data = pd.read_csv(test_file_path, skiprows=1, header=None)
    data = data.apply(normalize)

    sequences = []
    for i in range(len(data)):
        seq = data.iloc[i].values
        sequences.append(seq)

    sequences = np.array(sequences)
    sequences = torch.tensor(sequences, dtype=torch.float32)

    return sequences

test_file_path = "./for-testing/gesture_3__1739535951.csv"
sample_input = get_sample_input_from_file(test_file_path)
print(f"Prediction for testing file {test_file_path} = " + str(predict_gesture_by_sample_input(sample_input)))