import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os

csv_files = []

# 1. Data Loading and Preprocessing
for file_name in os.listdir("."):
    # Check if the file name starts with 'gesture_' and ends with '.csv'
    if file_name.startswith("gesture_1") and file_name.endswith(".csv"):
        csv_files.append(file_name)

csv_files.sort()

data = []
for file in csv_files:
    df = pd.read_csv(file, skiprows=1, header=None).values
    data.append(df)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = 0.5  # Your hardcoded target
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 52  # Adjust as needed
X = []
y = []

for d in data:
    X_seq, y_seq = create_sequences(d, sequence_length)
    X.extend(X_seq)
    y.extend(y_seq)

X = np.array(X)
y = np.array(y)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 4. Create DataLoaders (for batching)
batch_size = 32 # Adjust as needed
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle test data


# 5. Define the RNN Model (LSTM)
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True) # batch_first handles input shape (batch, seq, features)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)  # output (batch, seq_len, hidden_dim)
        output = output[:, -1, :]  # Get the last time step's output (batch, hidden_dim)
        output = self.fc(output)    # (batch, output_dim)
        return output

input_dim = X_train.shape[2] # Number of features
hidden_dim = 50 # Adjust as needed
output_dim = 1 # For regression
model = RNN(input_dim, hidden_dim, output_dim)

# 6. Define Loss Function and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adjust learning rate as needed

# 7. Training Loop
num_epochs = 20  # Adjust as needed

for epoch in range(num_epochs):
    for i, (X_batch, y_batch) in enumerate(train_loader):
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch) # .squeeze() to remove extra dimension

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 8. Prediction on the 2nd file
test_file_index = 1
test_data = pd.read_csv(csv_files[test_file_index], skiprows=1, header=None).values

X_test_file, _ = create_sequences(test_data, sequence_length)
X_test_file = torch.tensor(X_test_file, dtype=torch.float32)

model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradients for prediction
    predictions = model(X_test_file)

print("Predictions for the 2nd file:")
print(predictions)

# 9. Evaluation
model.eval()  # Set to evaluation mode
total_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Test Loss: {avg_loss:.4f}")

# 10. Save the model
torch.save(model.state_dict(), 'my_pytorch_rnn_model.pth')