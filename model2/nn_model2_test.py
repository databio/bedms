import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nn_model2_train import *
from nn_model2_train import trainer 
from nn_model2_preprocess import *
import pickle
import json 

def load_optimized_results():
    with open('hyperparam_optim_results.pkl' , 'rb') as f:
        optimized_results=pickle.load(f)
    return optimized_results

optimized_results=load_optimized_results()
best_hyperparameters = optimized_results['best_hyperparameters']
hidden_size = best_hyperparameters['hidden_size']
batch_size=best_hyperparameters['batch_size']
best_model_path="nn_model2_best.pth"  

best_model=headers_NN(input_size_values=X_test_tensor.shape[1],input_size_headers=X_test_headers_tensor.shape[1], hidden_size=hidden_size, output_size=len(label_encoder.classes_))
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

test_loader=DataLoader(TensorDataset(X_test_tensor, X_test_headers_tensor, y_test_tensor), batch_size=batch_size)

device=torch.device("cpu")
all_preds=[]
all_labels=[]
with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        input_values = X_test_tensor[i:i + batch_size].to(device)
        input_headers=X_test_headers_tensor[i:i+batch_size].to(device)
        labels = y_test_tensor[i:i + batch_size].to(device)

        outputs = best_model(input_values, input_headers)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Inverse transform labels
decoded_preds = label_encoder.inverse_transform(all_preds)
decoded_labels=label_encoder.inverse_transform(all_labels)

#test accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy Score{accuracy}")

num_categories=len(df_test)

grouped_preds = [decoded_preds[i:i+num_categories] for i in range(0, len(all_preds), num_categories)]
grouped_labels= [decoded_labels[i:i+num_categories] for i in range(0, len(all_labels), num_categories)]
consensus=[]
labels=[]
top_three_preds=[]
output_file_path="predictions_model2.json"

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

output_dict = {}
for label, consensus_values, top_three in zip(labels, consensus, top_three_preds):
    label_dict = {}
    for pred, percentage in top_three.items():
        label_dict[pred] = percentage
    output_dict[label] = label_dict

print(output_dict)

with open(output_file_path, "w") as f:
    json.dump(output_dict, f)

num_epochs=10
#generating confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(combined_labels), yticklabels=np.unique(combined_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.savefig("confusion_matrix_model2.jpg")
plt.show()

#plotting learning curve - accuracy
plt.plot(range(1, num_epochs + 1), trainer.train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), trainer.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("accuracy_model2.jpg")
plt.show()

#learning curve - loss
plt.plot(range(1, num_epochs + 1), trainer.train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), trainer.val_losses , label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("loss_model2.jpg")
plt.show()



