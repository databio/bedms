import pandas as pd
import numpy as np
import os
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from collections import Counter
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


output_file_path="../predictions.txt" #change path
process = psutil.Process(os.getpid())
#for memory usage
def print_memory_usage():
    print(f"Memory usage: {process.memory_info().rss / 2.**20} MB")

#reading input tsv
df=pd.read_csv("../encode_metadata.tsv", sep="\t") #change path 
df.replace('NA', np.nan, inplace=True) #preprocessing for imputations
#imputations
for column in df.columns:
    most_common_val=df[column].mode().iloc[0]
    df[column]=df[column].fillna(most_common_val)
#train test validation split 
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_test, df_val = train_test_split(df_temp, test_size=0.5, random_state=42)

#features and labels
X_train = [df_train[column].astype(str).tolist() for column in df_train.columns]
y_train = df_train.columns
X_test = [df_test[column].astype(str).tolist() for column in df_test.columns]
y_test = df_test.columns
X_val = [df_val[column].astype(str).tolist() for column in df_val.columns]
y_val = df_val.columns

#expanding the labels to match with the inputs (X)
y_train_expanded = [label for label in y_train for _ in range(len(df_train))]
y_test_expanded = [label for label in y_test for _ in range(len(df_test))]
y_val_expanded = [label for label in y_val for _ in range(len(df_val))]

#Flattening column values seuqneces
X_train_flat = [item for sublist in X_train for item in sublist]
X_test_flat = [item for sublist in X_test for item in sublist]
X_val_flat = [item for sublist in X_val for item in sublist]

#Reshaping to 2D arrays
X_train_2d = [[val] for val in X_train_flat]
X_test_2d = [[val] for val in X_test_flat]
X_val_2d = [[val] for val in X_val_flat]

# One-hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = enc.fit_transform(X_train_2d)
X_test_encoded = enc.transform(X_test_2d)
X_val_encoded = enc.transform(X_val_2d)
with open('v1_encoder.pkl', 'wb') as f:
    pickle.dump(enc, f)
print_memory_usage()

#label encoding
combined_labels = np.concatenate([y_train_expanded, y_test_expanded, y_val_expanded])  
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)
y_train_encoded = label_encoder.transform(y_train_expanded)
y_test_encoded = label_encoder.transform(y_test_expanded)
y_val_encoded = label_encoder.transform(y_val_expanded)
print_memory_usage()

#converting to tensors
X_train_tensor=torch.tensor(X_train_encoded.toarray(), dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_encoded.toarray(), dtype=torch.float32)
X_val_tensor=torch.tensor(X_val_encoded.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

print_memory_usage()

#NN model
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#optimizer, loss functiom
input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = len(np.unique(combined_labels))
model = NN(input_size, hidden_size, output_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

#move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#training the model
train_accuracies=[]
num_epochs = 10
batch_size=32
for epoch in range(num_epochs):
    model.train()
    total_correct_train=0
    total_samples_train=0
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i + batch_size].to(device)
        labels = y_train_tensor[i:i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    #accuracy chunk start
        _, predicted_train = torch.max(outputs, 1)
        correct_train = (predicted_train == labels).sum().item()
        total_correct_train+=correct_train
        total_samples_train+=batch_size
    train_accuracy = total_correct_train / total_samples_train
    train_accuracies.append(train_accuracy)
    #accuracy chunk end 
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
final_train_accuracy=train_accuracies[-1]
print(f"Final Training Accuracy:{final_train_accuracy}")

#testing the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    for i in range(0, len(X_val_tensor), batch_size):
        inputs = X_val_tensor[i:i + batch_size].to(device)
        labels = y_val_tensor[i:i + batch_size].to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_loss = loss_fn(test_outputs, y_test_tensor.to(device)).item()
print(f'Test Loss: {test_loss}')
#test accuracy
_, predicted = torch.max(test_outputs, 1)
_, top_predictions = torch.topk(test_outputs, k=3, dim=1) #top 3 preds
correct = (predicted == y_test_tensor.to(device)).sum().item()
test_accuracy = correct / len(X_test_tensor)
print(f'Test Accuracy: {test_accuracy}')

#save the model
torch.save(model.state_dict(), "../model_v4.pth")
torch.save(label_encoder.classes_, "../nn_label_encoder_v4.pth")

#consensus predictions
deencoded_labels = label_encoder.inverse_transform(all_labels)
deencoded_preds = label_encoder.inverse_transform(all_preds)

num_categories = len(df_test)
grouped_preds = [deencoded_preds[i:i+num_categories] for i in range(0, len(all_preds), num_categories)]
grouped_labels= [deencoded_labels[i:i+num_categories] for i in range(0, len(all_labels), num_categories)]
consensus=[]
labels=[]
top_three_preds=[]

with open(output_file_path,"w") as output_file:

    for i, category_preds in enumerate(grouped_preds):
        counts=Counter(category_preds)
        total_predictions=len(category_preds)
        consensus_value=max(counts, key=counts.get)
        consensus_percentage=counts[consensus_value]/total_predictions
        top_three_preds.append({pred: count / total_predictions for pred, count in counts.most_common(3)})
        consensus.append((consensus_value, consensus_percentage))
    for i, category_labels in enumerate(grouped_labels):
        counts=Counter(category_labels)
        most_popular_label=max(counts, key=counts.get)
        labels.append(most_popular_label)

    for i, (label, consensus_values) in enumerate(zip(labels, consensus)):
        output_file.write(f"{i+1}:{label}\n")
        output_file.write(f"{top_three_preds[i]}\n")
    print(f"Output printed to path {output_file_path}")
'''
# Generating confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(combined_labels), yticklabels=np.unique(combined_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("../confusion_matrix_v1.jpg")
plt.show()
'''
