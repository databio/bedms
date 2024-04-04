import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nn_model3_train import *
from nn_model3_train import trainer 
from nn_model3_preprocess import *
import json
import pickle 

def load_optimized_results():
    with open('hyperparam_optim_results.pkl' , 'rb') as f:
        optimized_results=pickle.load(f)
    return optimized_results

optimized_results=load_optimized_results()
best_hyperparameters = optimized_results['best_hyperparameters']
hidden_size = best_hyperparameters['hidden_size']
batch_size=best_hyperparameters['batch_size']
best_model_path="nn_model3_best.pth"  

output_file_path = "predictions_model3.json"

best_model=BoWModel(input_size_values=X_test_bow_tensor.shape[1], input_size_headers=X_test_header_bow_tensor.shape[1], hidden_size=hidden_size, output_size=len(np.unique(np.concatenate((y_train, y_val)))))

best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

test_loader=DataLoader(TensorDataset(X_test_bow_tensor, X_test_header_bow_tensor, y_test_tensor), batch_size=batch_size)

device=torch.device("cpu")
all_preds=[]
all_labels=[]

with torch.no_grad():
    for i in range(0, len(X_test_bow_tensor), batch_size):
        input_values = X_test_bow_tensor[i:i + batch_size].to(device)
        input_headers=X_test_header_bow_tensor[i:i+batch_size].to(device)
        labels = y_test_tensor[i:i+batch_size].to(device)

        outputs = best_model(input_values, input_headers)
        _, preds = torch.max(outputs, 1)
        _, preds_top3 = torch.topk(outputs, k=3, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#Inverse transform labels
deencoded_labels = label_encoder.inverse_transform(all_labels)
deencoded_preds = label_encoder.inverse_transform(all_preds)

#test accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy Score{accuracy}")

#printing to output dictionary as json
softmax_output = torch.softmax(outputs, dim=1).cpu().numpy()
output_dict={}
num_categories=len(df_test)
grouped_preds = [deencoded_preds[i:i+num_categories] for i in range(0, len(all_preds), num_categories)]
grouped_labels= [deencoded_labels[i:i+num_categories] for i in range(0, len(all_labels), num_categories)]

for i, attribute in enumerate(deencoded_labels):
    prob_scores = softmax_output[i]
    top_three_indices = np.argsort(prob_scores)[-3:][::-1]
    top_three_predictions = {}
    for idx in top_three_indices:
        decoded_pred = label_encoder.classes_[idx]
        top_three_predictions[decoded_pred] = prob_scores[idx]
    output_dict[attribute] = top_three_predictions

print(output_dict)

#save to file
output_dict_converted = {}
for key, value in output_dict.items():
    output_dict_converted[key] = {k: float(v) for k, v in value.items()}
with open(output_file_path, "w") as f:
    json.dump(output_dict_converted, f)


num_epochs=10
#generating the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(combined_labels), yticklabels=np.unique(combined_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.savefig("confusion _matrix_model3.jpg")
plt.show()

#plotting learning curve - accuracy
plt.plot(range(1, num_epochs + 1), trainer.train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), trainer.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("accuracy_model3.jpg")
plt.show()

#learning curve - loss
plt.plot(range(1, num_epochs + 1), trainer.train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), trainer.val_losses , label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("loss_model3.jpg")
plt.show()
