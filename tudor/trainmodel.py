import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import joblib
import json
import os


# --------- Init Params -------
train_files = glob.glob("tudor/samples2/raw_normalized/*.csv")
#Model params
hidden_size = 64
num_layers = 3
epochs = 1000
modelName = 'raw_normalized64_3'
saveModelFileName = f"tudor/model/{modelName}/gru_model.pth"
saveScalerFileName = f"tudor/model/{modelName}/scaler.pkl"
modelMetadataFile = f"tudor/model/{modelName}/info_model.txt"

# ---------------------
os.makedirs(f'tudor/model/{modelName}', exist_ok=True)

# Define GRU Model
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out

# Custom Dataset
class GestureDataset(Dataset):
    def __init__(self, file_paths, label_encoder, scaler, train=True):
        self.data = []
        self.labels = []
        self.train = train
        for file_path in file_paths:
            df = pd.read_csv(file_path, skiprows=1)  # Ignore first row
            features = df.values
            if train:
                scaler.partial_fit(features)  # Fit scaler on training data
            features = scaler.transform(features)  # Normalize
            self.data.append(features)
            if train:
                gesture_name = os.path.basename(file_path).split('_')[1]  # Extract label from file name
                self.labels.append(label_encoder.transform([gesture_name])[0])
        self.data = [torch.tensor(d, dtype=torch.float32) for d in self.data]
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx], [] if self.train == False else self.labels[idx]
    
def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unzip batch
    sequences_padded = pad_sequence(sequences, batch_first=True)  # Pad sequences
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels

# Load and process data

labels = sorted(set(os.path.basename(f).split('_')[1] for f in train_files))
print(f'Labels found: {labels}')
label_encoder = LabelEncoder()
label_encoder.fit(labels)
scaler = StandardScaler()

train_dataset = GestureDataset(train_files, label_encoder, scaler, train=True)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)


# Model parameters
input_size = next(iter(train_dataset))[0].shape[1]  # Feature size
output_size = len(labels)


model = GRUNet(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets = targets
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    if total_loss/len(train_loader) < 0.01:
        break
torch.save(model.state_dict(), saveModelFileName)
joblib.dump(scaler, saveScalerFileName)
metadata = {
    'modelFile': saveModelFileName,
    'scalerFile': saveScalerFileName,
    'hidden_size' : hidden_size,
    "num_layers" : num_layers,
    "labels": labels,
    "input_size": input_size,
    "output_size": output_size
}
with open(modelMetadataFile, "w") as file:
    json.dump(metadata, file, indent=4)
    print(f"MetaData saved successfully! to file {file}")
print("Model saved successfully!")
# # Evaluation
# model.eval()
# with torch.no_grad():
#     for inputs, _ in test_loader:
#         # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
#         outputs = model(inputs)
#         # Apply softmax to get probabilities
#         probabilities = F.softmax(outputs, dim=1)
#          # Get predicted class and confidence
#         predicted_class = torch.argmax(probabilities, dim=1)
#         confidence = torch.max(probabilities, dim=1).values


#         predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
#         print(f"Predicted Gesture: {predicted_label}, Confidence: {confidence.item():.4f}")