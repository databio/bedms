import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nn_model4_train import *
from nn_model4_train import trainer 
from nn_model4_preprocess import *
import json

output_file_path = "predictions.json"
model=BoWSTModel(input_size_values=X_test_bow_tensor.shape[1], input_size_headers=X_test_headers_tensor.shape[1], hidden_size=64, output_size=len(np.unique(y_train)))
model.load_state_dict(torch.load(model_path))
model.eval()
batch_size=32
test_loader=DataLoader(TensorDataset(X_test_bow_tensor, X_test_headers_tensor, y_test_tensor), batch_size=batch_size)
device=torch.device("cpu")
all_preds=[]
all_labels=[]

with torch.no_grad():
    for i in range(0, len(X_test_bow_tensor), batch_size):
        input_values = X_test_bow_tensor[i:i + batch_size].to(device)
        input_headers=X_test_headers_tensor[i:i+batch_size].to(device)
        labels = y_test_tensor[i:i+batch_size].to(device)

        outputs = model(input_values, input_headers)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#Inverse tranform labels
deencoded_labels = label_encoder.inverse_transform(all_labels)
deencoded_preds = label_encoder.inverse_transform(all_preds)

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
#plt.savefig("confusion _matrix_model4.jpg")
plt.show()

#plotting learning curve - accuracy
plt.plot(range(1, num_epochs + 1), trainer.train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), trainer.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("accuracy_model4.jpg")
plt.show()

#learning curve - loss
plt.plot(range(1, num_epochs + 1), trainer.train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), trainer.val_losses , label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("loss_model4.jpg")
plt.show()


