# This model trains on both values and headers. Each column is a single bag of words in this model.
# TODO Add comments, parameter count, optuna, saved model as state_dict
# TODO Change the input to multiple tables as input for the model. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging 

#logging set up
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#NN model
class BoWModel(nn.Module):
    def __init__(self, input_size_values, input_size_headers, hidden_size, output_size):
        super(BoWModel, self).__init__()
        self.fc_values = nn.Linear(input_size_values, hidden_size)
        self.fc_headers = nn.Linear(input_size_headers, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x_values, x_headers):
        x_values = F.relu(self.fc_values(x_values))
        x_headers = F.relu(self.fc_headers(x_headers))
        
        x_combined = torch.cat((x_values, x_headers), dim=1)

        x_combined = self.fc_combined(x_combined)
        return x_combined
    
def load(file_path):
    df=pd.read_csv(file_path, sep=",")
    df.replace('NA', np.nan, inplace=True)

    #subset for experimentation
    subset_df=df.head(5000)
    df=subset_df

    for column in df.columns:
        most_common_val = df[column].mode().iloc[0]
        df[column] = df[column].fillna(most_common_val)
    return df 

def data_split(df_values, df_headers):
    df_values_train, df_values_temp=train_test_split(df_values, test_size=0.2, random_state=42)
    df_values_test,df_values_val=train_test_split(df_values_temp, test_size=0.5, random_state=42)

    df_headers_train, df_headers_temp=train_test_split(df_headers, test_size=0.2, random_state=42)
    df_headers_test, df_headers_val=train_test_split(df_headers_temp, test_size=0.5, random_state=42)

    # BELOW PART ONLY FOR TESTING ON UNSEEN DATASET
    #df_headers_test=pd.read_csv("../data/encode_headers_test.tsv", sep="\t")
    #df_values_test=pd.read_csv("../data/encode_values_test.tsv", sep="\t")
    #COMMENT OUT ABOVE STATEMENT TODO

    X_values_train = [df_values_train[column].astype(str).tolist() for column in df_values_train.columns]
    X_values_test = [df_values_test[column].astype(str).tolist() for column in df_values_test.columns]
    X_values_val = [df_values_val[column].astype(str).tolist() for column in df_values_val.columns]

    X_headers_train=[df_headers_train[column].astype(str).tolist() for column in df_headers_train.columns]
    X_headers_test=[df_headers_test[column].astype(str).tolist() for column in df_headers_test.columns]
    X_headers_val=[df_headers_val[column].astype(str).tolist() for column in df_headers_val.columns]

    y_train=df_headers_train.columns
    y_test=df_headers_test.columns
    y_val=df_headers_val.columns

    #forming a single string - bag of words
    X_values_train_strings = [" ".join([str(val) for val in sublist]) for sublist in X_values_train]
    X_values_test_strings = [" ".join([str(val) for val in sublist]) for sublist in X_values_test]
    X_values_val_strings = [" ".join([str(val) for val in sublist]) for sublist in X_values_val]

    X_headers_train_strings=[" ".join([str(val) for val in sublist]) for sublist in X_headers_train]
    X_headers_test_strings=[" ".join([str(val) for val in sublist]) for sublist in X_headers_test]
    X_headers_val_strings=[" ".join([str(val) for val in sublist]) for sublist in X_headers_val]


    X_values_train=X_values_train_strings
    X_values_test= X_values_test_strings
    X_values_val = X_values_val_strings
    X_headers_train=X_headers_train_strings
    X_headers_test=X_headers_test_strings
    X_headers_val=X_headers_val_strings

    return X_values_train, X_values_test, X_values_val, \
          X_headers_train, X_headers_test, X_headers_val, \
        y_train, y_test, y_val 

def encoding(X_values_train, X_values_test, X_values_val, \
        X_headers_train, X_headers_test, X_headers_val, \
        y_train, y_test, y_val):
    vectorizer = CountVectorizer()
    all_data=X_values_train+X_headers_train
    vectorizer.fit(all_data)
    X_values_train_bow= vectorizer.transform(X_values_train)
    X_values_test_bow= vectorizer.transform(X_values_test)
    X_values_val_bow= vectorizer.transform(X_values_val)

    X_headers_train_bow = vectorizer.transform(X_headers_train)
    X_headers_test_bow = vectorizer.transform(X_headers_test)
    X_headers_val_bow = vectorizer.transform(X_headers_val)

    combined_labels = np.concatenate([y_train, y_test, y_val])
    label_encoder=LabelEncoder()
    combined_labels = np.concatenate([y_train, y_test, y_val])
    label_encoder.fit(combined_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    return X_values_train_bow, X_values_test_bow, X_values_val_bow, \
          X_headers_train_bow, X_headers_test_bow, X_headers_val_bow,\
              y_train_encoded, y_test_encoded, y_val_encoded, label_encoder, combined_labels
    
def to_tensor(X_values_train_bow, X_values_test_bow, X_values_val_bow, \
        X_headers_train_bow, X_headers_test_bow, X_headers_val_bow,\
        y_train_encoded, y_test_encoded, y_val_encoded):
    X_values_train_tensor=torch.tensor(X_values_train_bow.toarray(), dtype=torch.float32)
    X_values_test_tensor=torch.tensor(X_values_test_bow.toarray(), dtype=torch.float32)
    X_values_val_tensor=torch.tensor(X_values_val_bow.toarray(), dtype=torch.float32)

    X_headers_train_tensor=torch.tensor(X_headers_train_bow.toarray(), dtype=torch.float32)
    X_headers_test_tensor=torch.tensor(X_headers_test_bow.toarray(), dtype=torch.float32)
    X_headers_val_tensor=torch.tensor(X_headers_val_bow.toarray(), dtype=torch.float32)

    y_train_tensor=torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor=torch.tensor(y_test_encoded, dtype=torch.long)
    y_val_tensor=torch.tensor(y_val_encoded, dtype=torch.long)

    return X_values_train_tensor, X_values_test_tensor, X_values_val_tensor , \
        X_headers_train_tensor , X_headers_test_tensor, X_headers_val_tensor, \
            y_train_tensor, y_test_tensor, y_val_tensor
    
def model_training_validation(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, output_size):
    train_accuracies=[]
    train_losses=[]
    val_accuracies=[]
    val_losses=[]
    for epoch in range(num_epochs):
        model.train()
        total_loss_train=0.0
        total_correct_train=0
        total_samples_train=0
        for values_batch, headers_batch, labels in train_loader:
            optimizer.zero_grad()
            outputs=model(values_batch, headers_batch)
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train+=loss.item()
            _, predicted_train = torch.max(outputs, 1)
            correct_train = (predicted_train == labels).sum().item()
            total_correct_train+=correct_train
            total_samples_train+=labels.size(0)
        train_accuracy = total_correct_train / total_samples_train
        train_accuracies.append(train_accuracy)
        train_loss = total_loss_train / len(X_values_train_tensor)
        train_losses.append(train_loss)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Training Accuracy:{train_accuracy}')

        model.eval()
        total_loss_val=0.0
        total_correct_val=0
        total_samples_val=0
        with torch.no_grad():
            y_score = model(X_values_val_tensor, X_headers_val_tensor).cpu().numpy()
            for values_batch, headers_batch, labels in val_loader:
                outputs=model(values_batch, headers_batch)
                loss=loss_fn(outputs, labels)
                total_loss_val=loss.item()
                _, predicted_val = torch.max(outputs, 1)
                correct_val = (predicted_val == labels).sum().item()
                total_correct_val+=correct_val
                total_samples_val+=labels.size(0)
        validation_accuracy = total_correct_val / total_samples_val
        val_accuracies.append(validation_accuracy)
        val_loss = total_loss_val / len(X_values_val_tensor)
        val_losses.append(val_loss)
    final_train_accuracy=train_accuracies[-1]
    final_val_accuracy=val_accuracies[-1]
    final_train_loss=train_losses[-1]
    final_val_loss=val_losses[-1]
    logger.info(f"Final Training Accuracy:{final_train_accuracy}, Final Training Loss:{final_train_loss}")
    logger.info(f"Final Validation Accuracy:{final_val_accuracy}, Final Validation Loss:{final_val_loss}")
    #calculating roc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(output_size): 
        fpr[i], tpr[i], _ = roc_curve((y_val_tensor == i).cpu(), y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc

def model_testing(model, test_loader, loss_fn):
    all_preds=[]
    all_labels=[]
    model.eval()
    total_loss_test=0.0
    total_correct_test=0
    total_samples_test=0
    with torch.no_grad():
        for values_batch, headers_batch, labels in test_loader:
            outputs=model(values_batch, headers_batch)
            loss=loss_fn(outputs, labels)
            total_loss_test+=loss.item()
            _, predicted_test = torch.max(outputs, 1)
            correct_test = (predicted_test == labels).sum().item()
            total_correct_test += correct_test
            total_samples_test += labels.size(0)
            all_preds.extend(predicted_test.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_accuracy=total_correct_test/total_samples_test
    test_loss=total_loss_test/len(test_loader)
    logger.info(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
    return all_preds, all_labels, outputs

def json_output(all_labels, all_preds, label_encoder, outputs, num_categories, output_file_path):
    deencoded_labels = label_encoder.inverse_transform(all_labels)
    deencoded_preds = label_encoder.inverse_transform(all_preds)

    softmax_output = torch.softmax(outputs, dim=1).cpu().numpy()
    output_dict={}
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

def confusion_mat(all_labels, all_preds, combined_labels):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(combined_labels), yticklabels=np.unique(combined_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix_model3.jpg")
    plt.show()
    # Calculate class-wise accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    # Print class-wise accuracy
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {i}: {acc:.4f}")

def learning_curve(num_epochs, train_accuracies, val_accuracies, train_losses, val_losses):
    #accuracy
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_model3.jpg")
    plt.show()
    #loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses , label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_model3.jpg")
    plt.show()

def auc_roc_curve(fpr,tpr, roc_auc, output_size):
    plt.figure(figsize=(12, 12))
    for i in range(output_size):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d) (AUC = %0.2f)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_model3.jpg")
    plt.show()

if __name__=="__main__":
    values_file_path="../data/encode_subset_metadata.csv"
    headers_file_path="../data/encode_subset_headers.csv"
    #load the data into values and headers
    df_values=load(values_file_path)
    df_headers=load(headers_file_path)
    #train, test, validation split
    X_values_train, X_values_test, X_values_val, \
        X_headers_train, X_headers_test, X_headers_val, \
        y_train, y_test, y_val= \
        data_split(df_values, df_headers)
    X_values_train_bow, X_values_test_bow, X_values_val_bow, \
        X_headers_train_bow, X_headers_test_bow, X_headers_val_bow,\
        y_train_encoded, y_test_encoded, y_val_encoded, label_encoder, combined_labels=encoding(X_values_train, X_values_test, X_values_val, \
        X_headers_train, X_headers_test, X_headers_val, \
        y_train, y_test, y_val)
    X_values_train_tensor, X_values_test_tensor, X_values_val_tensor , \
        X_headers_train_tensor , X_headers_test_tensor, X_headers_val_tensor, \
            y_train_tensor, y_test_tensor, y_val_tensor=to_tensor(X_values_train_bow, X_values_test_bow, X_values_val_bow, \
        X_headers_train_bow, X_headers_test_bow, X_headers_val_bow,\
        y_train_encoded, y_test_encoded, y_val_encoded)
    #defining model parameters
    input_size_values=X_values_train_tensor.shape[1]
    input_size_headers=X_headers_train_tensor.shape[1]
    hidden_size=32
    output_size=len(label_encoder.classes_)
    #Instantiating the model 
    model=BoWModel(input_size_values, input_size_headers, hidden_size, output_size)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=0.001)

    train_data=TensorDataset(X_values_train_tensor, X_headers_train_tensor, y_train_tensor)
    train_loader=DataLoader(train_data, batch_size=32, shuffle=True)

    val_data=TensorDataset(X_values_val_tensor, X_headers_val_tensor, y_val_tensor)
    val_loader=DataLoader(val_data, batch_size=32)

    test_data=TensorDataset(X_values_test_tensor, X_headers_test_tensor, y_test_tensor)
    test_loader=DataLoader(test_data, batch_size=32)
    #training and validation of the model
    num_epochs=10
    train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc= \
        model_training_validation(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, output_size)
    all_preds, all_labels, outputs=model_testing(model, test_loader, loss_fn)
    #json output 
    num_categories=len(label_encoder.classes_)
    output_file_path="predictions_model3.json"
    json_output(all_labels, all_preds, label_encoder, outputs, num_categories, output_file_path)
    #calculating preciison, recall, f1
    precision=precision_score(all_labels, all_preds, average="macro")
    recall=recall_score(all_labels, all_preds, average="macro")
    f1=f1_score(all_labels, all_preds, average="macro")
    logger.info(f"Precision:{precision}, Recall: {recall}, F1 Score: {f1}")
    #generating confusion matrix
    confusion_mat(all_labels, all_preds, combined_labels)
    #generating learning curves - accuracy and loss
    learning_curve(num_epochs, train_accuracies, val_accuracies, train_losses, val_losses)
    #generating AUC ROC 
    auc_roc_curve(fpr,tpr, roc_auc, output_size)
    
    