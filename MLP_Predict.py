import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    max_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data loading

# The x construction of the graph
embd = pd.read_csv(
    'The semantic segmentation features of each road after processing.csv')  # Road semantic segmentation feature reading, a total of 5075 roads
features = []
for i in range(5075):
    try:
        e = embd[(embd['RoadID']) == i + 1].values[0][2:21].tolist()
        features.append(e)
    except:
        features.append([0] * 19)  # Part of the road has no features and is filled with zeros

# The y construction of the graph
data_y_org = pd.read_csv('CGrade.csv')  # CO emissions read, 5,075 roads
labels = data_y_org['TotalBreak'].values.tolist()
for t in range(5075):
    if math.isnan(labels[t]) or labels[t] == 0:
        labels[t] = 1  # If it is an empty nan, it is filled with 0

# Data standardization
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert the label to an integer type, making sure to start at 0
labels = [int(label) - 1 for label in labels]

# Divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to PyTorch Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # Use Long Tensor to represent a label of type integer
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)


# Define the neural network model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the model and optimizer
input_size = 19
hidden_size1 = 64
hidden_size2 = 32
output_size = 7

model = MLP(input_size, hidden_size1, hidden_size2, output_size)
# Define loss functions and optimizers
criterion = nn.CrossEntropyLoss()  # Use cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training model
num_epochs = 3000

for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        # Calculate the accuracy on the training set
        model.eval()
        with torch.no_grad():
            train_predicted_labels = model(X_train_tensor)
            _, train_predicted_classes = torch.max(train_predicted_labels, 1)
            train_accuracy = accuracy_score(y_train, train_predicted_classes.numpy())

        # Calculate additional performance metrics
        train_mse = mean_squared_error(y_train, train_predicted_classes.numpy())
        train_rmse = math.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, train_predicted_classes.numpy())

        # Calculate MAPE, ME, MSL and other indicators
        train_pred_cpu = train_predicted_classes.cpu().numpy()
        train_labels_cpu = np.array(y_train)

        train_me = max_error(train_pred_cpu, train_labels_cpu)
        train_msl = mean_squared_log_error(train_pred_cpu, train_labels_cpu)

        # Trim the predicted values to avoid division by zero
        train_pred_cpu = np.clip(train_pred_cpu, 1, None)

        train_mape = mean_absolute_percentage_error(train_pred_cpu, train_labels_cpu)


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, MSE: {train_mse:.4f}, '
              f'RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, '
              f'MAPE: {train_mape:.4f}, ME: {train_me:.4f}, MSL: {train_msl:.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    predicted_labels = model(X_test_tensor)
    _, predicted_classes = torch.max(predicted_labels, 1)  # Select the category with the highest prediction probability as the prediction

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, predicted_classes.numpy())

    # Calculate additional performance metrics
    test_mse = mean_squared_error(y_test, predicted_classes.numpy())
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, predicted_classes.numpy())

    # Calculate MAPE, ME, MSL and other indicators
    test_pred_cpu = predicted_classes.cpu().numpy()
    test_labels_cpu = np.array(y_test)

    test_me = max_error(test_pred_cpu, test_labels_cpu)
    test_msl = mean_squared_log_error(test_pred_cpu, test_labels_cpu)

    # Trim the predicted values to avoid division by zero
    test_pred_cpu = np.clip(test_pred_cpu, 1, None)

    test_mape = mean_absolute_percentage_error(test_pred_cpu, test_labels_cpu)


    print(f'Test Accuracy: {test_accuracy:.4f}, MSE: {test_mse:.4f}, '
          f'RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, '
          f'MAPE: {test_mape:.4f}, ME: {test_me:.4f}, MSL: {test_msl:.4f}')
