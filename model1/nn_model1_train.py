#This model only gets trained on column values
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from nn_model1_preprocess import *

model_path="nn1_model.pth"
encoder_path="encoder_model1.pth"

class NN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN1, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ModelTraining:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.model = NN1(input_size, hidden_size, output_size)
        self.label_encoder = label_encoder
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_accuracies=[]
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []
        self.num_epochs=10

    def train(self, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs, batch_size, device='cpu'):
        
        self.model=self.model.to(device)

        #Training starts
        for epoch in range(num_epochs):
            self.model.train()
            total_correct_train = 0
            total_samples_train = 0
            total_loss_train = 0.0
            for i in range(0, len(X_train_tensor), batch_size):
                inputs = X_train_tensor[i:i + batch_size].to(device)
                labels = y_train_tensor[i:i + batch_size].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted_train = torch.max(outputs, 1)
                correct_train = (predicted_train == labels).sum().item()
                total_correct_train += correct_train
                total_samples_train += batch_size
                total_loss_train += loss.item() * inputs.size(0)
            train_accuracy = total_correct_train / total_samples_train
            self.train_losses.append(total_loss_train / len(X_train_tensor))
            self.train_accuracies.append(train_accuracy)

            # Validation starts
            self.model.eval()
            total_correct_val = 0
            total_samples_val = 0
            total_loss_val = 0.0
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), batch_size):
                    inputs = X_val_tensor[i:i + batch_size].to(device)
                    labels = y_val_tensor[i:i + batch_size].to(device)

                    outputs = self.model(inputs)
                    _, predicted_val = torch.max(outputs, 1)
                    correct_val = (predicted_val == labels).sum().item()
                    total_correct_val += correct_val
                    total_samples_val += batch_size
                    total_loss_val += loss.item() * inputs.size(0)
            val_accuracy = total_correct_val / total_samples_val
            self.val_losses.append(total_loss_val / len(X_val_tensor))
            self.val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')

        return self.train_accuracies, self.val_accuracies, self.train_losses, self.val_losses

    def save_model(self, model_path,encoder_path):
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.label_encoder.classes_, encoder_path)

input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = len(np.unique(combined_labels))
start_time_train=time.time()
trainer = ModelTraining(input_size, hidden_size, output_size, learning_rate=0.05)
trainer.train(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=10, batch_size=32, device='cpu')
end_time_train=time.time()
trainer.save_model(model_path,encoder_path)
print("Model Training Done.")
time_taken=end_time_train-start_time_train
print(f"Total time taken for training:{time_taken:.2f} seconds")


