import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nn_model1_train import *
from nn_model1_train import trainer 
from nn_model1_preprocess import *

    
model=NN1(input_size=X_test_tensor.shape[1], hidden_size=64, output_size=len(label_encoder.classes_))
model.load_state_dict(torch.load(model_path))
model.eval()

batch_size=32
test_loader=DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

all_preds = []
all_labels=[]
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Inverse transform labels
decoded_preds = label_encoder.inverse_transform(all_preds)
decoded_labels=label_encoder.inverse_transform(all_labels)


num_categories=len(df_test)

grouped_preds = [decoded_preds[i:i+num_categories] for i in range(0, len(all_preds), num_categories)]
grouped_labels= [decoded_labels[i:i+num_categories] for i in range(0, len(all_labels), num_categories)]
consensus=[]
labels=[]
top_three_preds=[]
output_file_path="predictions_model1.txt"
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

    output_dict = {}
    for label, consensus_values, top_three in zip(labels, consensus, top_three_preds):
        label_dict = {}
        for pred, percentage in top_three.items():
            label_dict[pred] = percentage
        output_dict[label] = label_dict
    output_file.write(str(output_dict))
print(output_dict)


num_epochs=10
#confusion matrix 
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(combined_labels), yticklabels=np.unique(combined_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.savefig("confusion_matrix_model1.jpg")
plt.show()

#learning curve - accuracy
plt.plot(range(1, num_epochs + 1), trainer.train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), trainer.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("accuracy_model1.jpg")
plt.show()

#learning curve plotting - loss
plt.plot(range(1, num_epochs + 1), trainer.train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), trainer.val_losses , label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
#plt.savefig("loss_model1.jpg")
plt.show()


