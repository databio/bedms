import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from nn_model5_preprocess import *

model_path="nn_model5.pth"
encoder_path="encoder_model5.pth"

#NN model - sentence transformers 
class sentence_transformer_NN(nn.Module):
    def __init__(self, input_size_values, input_size_headers, hidden_size, output_size):
        super(sentence_transformer_NN, self).__init__()
        self.fc_values = nn.Linear(input_size_values, hidden_size)
        self.fc_headers = nn.Linear(input_size_headers, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x_values, x_headers):
        x_values = F.relu(self.fc_values(x_values))
        x_headers = F.relu(self.fc_headers(x_headers))
        x_combined = torch.cat((x_values, x_headers), dim=1)
        x_combined = self.fc_combined(x_combined)
        return x_combined
    
class ModelTraining:
    def __init__(self, input_size_values, input_size_headers, hidden_size, output_size, learning_rate):
        self.model=sentence_transformer_NN(input_size_values, input_size_headers, hidden_size, output_size)
        self.label_encoder=label_encoder
        self.loss_fn=nn.CrossEntropyLoss()
        self.optimizer=optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_accuracies=[]
        self.val_accuracies=[]
        self.train_losses=[]
        self.val_losses=[]
    
    def train(self, X_train_tensor, X_train_headers_tensor, y_train_tensor, X_val_tensor, X_val_headers_tensor, y_val_tensor, num_epochs, batch_size, device='cpu'):
        self.model=self.model.to(device)
        
        for epoch in range(num_epochs):
            #training starts
            self.model.train()
            total_correct_train=0
            total_samples_train=0
            total_loss_train=0.0
            for i in range(0, len(X_train_tensor), batch_size):
                input_values = X_train_tensor[i:i + batch_size].to(device)
                input_headers=X_train_headers_tensor[i:i+batch_size].to(device)
                labels = y_train_tensor[i:i + batch_size].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_values, input_headers)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted_train = torch.max(outputs, 1)
                correct_train = (predicted_train == labels).sum().item()
                total_correct_train+=correct_train
                total_samples_train+=batch_size
                total_loss_train += loss.item()

            train_accuracy = total_correct_train / total_samples_train
            self.train_accuracies.append(train_accuracy)
            train_loss = total_loss_train / len(X_train_tensor)
            self.train_losses.append(train_loss)

            #validation starts
            self.model.eval()
            total_correct_val=0
            total_samples_val=0
            total_loss_val=0.0
            with torch.no_grad():
                y_score = self.model(X_val_tensor.to(device), X_val_headers_tensor.to(device)).cpu().numpy()
                for i in range(0, len(X_val_tensor), batch_size):
                    input_values = X_val_tensor[i:i + batch_size].to(device)
                    input_headers = X_val_headers_tensor[i:i+batch_size].to(device)
                    labels = y_val_tensor[i:i + batch_size].to(device)

                    outputs = self.model(input_values, input_headers)
                    loss=self.loss_fn(outputs,labels)
                    _, predicted_val = torch.max(outputs, 1)
                    correct_val = (predicted_val == labels).sum().item()
                    total_correct_val+=correct_val
                    total_samples_val+=labels.size(0)
                    total_loss_val+=loss.item() * labels.size(0)
            validation_accuracy = total_correct_val / total_samples_val
            self.val_accuracies.append(validation_accuracy)
            val_loss = total_loss_val / len(X_val_tensor)
            self.val_losses.append(val_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Validation Accuracy:{validation_accuracy}')
        
        final_train_accuracy=self.train_accuracies[-1]
        final_val_accuracy=self.val_accuracies[-1]
        final_train_loss = self.train_losses[-1]
        final_val_loss = self.val_losses[-1]
        print(f"Final Training Accuracy:{final_train_accuracy}, Final Training Loss:{final_train_loss}")
        print(f"Final Validation Accuracy:{final_val_accuracy}, Final Validation Loss:{final_val_loss}")

        return self.train_accuracies, self.val_accuracies, self.train_losses, self.val_losses
    
    def save_model(self, model_path, encoder_path):
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.label_encoder.classes_, encoder_path)

input_size_values = X_train_tensor.shape[1]
input_size_headers = X_train_headers_tensor.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train_expanded))

trainer=ModelTraining(input_size_values, input_size_headers, hidden_size, output_size, learning_rate=0.05)
trainer.train(X_train_tensor, X_train_headers_tensor, y_train_tensor, X_val_tensor, X_val_headers_tensor, y_val_tensor, num_epochs=10, batch_size=32, device='cpu')
trainer.save_model(model_path,encoder_path)
print("Model Training Done.")
    
