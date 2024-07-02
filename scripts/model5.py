# This model trains on Bag of Words values and Sentence Transformers headers
# TODO Add comments, parameter count, optuna, saved model as state_dict


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle

# logging set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NN model - sentence transformers


class sentence_transformer_NN(nn.Module):
    def __init__(
        self,
        input_size_values,
        input_size_headers,
        hidden_size,
        output_size,
        dropout_prob,
    ):
        super(sentence_transformer_NN, self).__init__()
        self.fc_values1 = nn.Linear(input_size_values, hidden_size)
        self.dropout_values1 = nn.Dropout(dropout_prob)
        self.fc_values2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_values2 = nn.Dropout(dropout_prob)
        self.fc_headers1 = nn.Linear(input_size_headers, hidden_size)
        self.dropout_headers1 = nn.Dropout(dropout_prob)
        self.fc_headers2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_headers2 = nn.Dropout(dropout_prob)
        self.fc_combined1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_combined1 = nn.Dropout(dropout_prob)
        self.fc_combined2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_values, x_headers):
        x_values = F.relu(self.fc_values1(x_values))
        x_values = self.dropout_values1(x_values)
        x_values = F.relu(self.fc_values2(x_values))
        x_values = self.dropout_values2(x_values)
        x_headers = F.relu(self.fc_headers1(x_headers))
        x_headers = self.dropout_headers1(x_headers)
        x_headers = F.relu(self.fc_headers2(x_headers))
        x_headers = self.dropout_headers2(x_headers)
        x_combined = torch.cat((x_values, x_headers), dim=1)
        x_combined = F.relu(self.fc_combined1(x_combined))
        x_combined = self.dropout_combined1(x_combined)
        x_combined = self.fc_combined2(x_combined)
        return x_combined


def load(file_path):
    df = pd.read_csv(file_path, sep=",", dtype=str)
    df.replace("NA", np.nan, inplace=True)

    for column in df.columns:
        if df[column].notna().any():  # skipping those columns that are all empty
            most_common_val = df[column].mode().iloc[0]
            # print(most_common_val)
            df[column] = df[column].fillna(most_common_val)
    return df


def data_split(df_values, df_headers):
    df_values_train, df_values_temp = train_test_split(
        df_values, test_size=0.2, random_state=42
    )
    df_values_test, df_values_val = train_test_split(
        df_values_temp, test_size=0.5, random_state=42
    )

    df_headers_train, df_headers_temp = train_test_split(
        df_headers, test_size=0.2, random_state=42
    )
    df_headers_test, df_headers_val = train_test_split(
        df_headers_temp, test_size=0.5, random_state=42
    )
    df_values_train = df_values
    df_headers_train = df_headers

    # BELOW PART ONLY FOR TESTING ON UNSEEN DATASET
    # df_headers_test=pd.read_csv("data/dummy_1_header.tsv", sep="\t")
    # df_values_test=pd.read_csv("data/dummy_1.tsv", sep="\t")
    # COMMENT OUT ABOVE STATEMENT TODO

    X_values_train = [
        df_values_train[column].astype(str).tolist()
        for column in df_values_train.columns
    ]
    X_values_test = [
        df_values_test[column].astype(str).tolist() for column in df_values_test.columns
    ]
    X_values_val = [
        df_values_val[column].astype(str).tolist() for column in df_values_val.columns
    ]

    X_headers_train = [
        df_headers_train[column].astype(str).tolist()
        for column in df_headers_train.columns
    ]
    X_headers_test = [
        df_headers_test[column].astype(str).tolist()
        for column in df_headers_test.columns
    ]
    X_headers_val = [
        df_headers_val[column].astype(str).tolist() for column in df_headers_val.columns
    ]

    y_train = df_headers_train.columns
    y_test = df_headers_train.columns
    y_val = df_headers_val.columns

    X_values_train_flat = [item for sublist in X_values_train for item in sublist]
    X_values_test_flat = [item for sublist in X_values_test for item in sublist]
    X_values_val_flat = [item for sublist in X_values_val for item in sublist]

    X_headers_train_flat = [item for sublist in X_headers_train for item in sublist]
    X_headers_test_flat = [item for sublist in X_headers_test for item in sublist]
    X_headers_val_flat = [item for sublist in X_headers_val for item in sublist]

    y_train_expanded = [label for label in y_train for _ in range(len(df_values_train))]
    y_test_expanded = [label for label in y_train for _ in range(len(df_values_test))]
    y_val_expanded = [label for label in y_train for _ in range(len(df_values_val))]

    X_values_train = X_values_train_flat
    X_values_test = X_values_test_flat
    X_values_val = X_values_val_flat
    X_headers_train = X_headers_train_flat
    X_headers_test = X_headers_test_flat
    X_headers_val = X_headers_val_flat
    y_train = y_train_expanded
    y_test = y_test_expanded
    y_val = y_val_expanded

    return (
        X_values_train,
        X_values_test,
        X_values_val,
        X_headers_train,
        X_headers_test,
        X_headers_val,
        y_train,
        y_test,
        y_val,
    )


def encoding(
    X_values_train,
    X_values_test,
    X_values_val,
    X_headers_train,
    X_headers_test,
    X_headers_val,
    y_train,
    y_test,
    y_val,
):

    model_name = "all-MiniLM-L6-v2"
    sentence_encoder = SentenceTransformer(model_name)

    X_values_train_embeddings = sentence_encoder.encode(X_values_train)
    X_values_test_embeddings = sentence_encoder.encode(X_values_test)
    X_values_val_embeddings = sentence_encoder.encode(X_values_val)

    X_headers_train_embeddings = sentence_encoder.encode(X_headers_train)
    X_headers_test_embeddings = sentence_encoder.encode(X_headers_test)
    X_headers_val_embeddings = sentence_encoder.encode(X_headers_val)
    """
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

    X_values_train_tokens = tokenizer(X_values_train, padding=True, truncation=True, return_tensors="pt")
    X_values_test_tokens = tokenizer(X_values_test, padding=True, truncation=True, return_tensors="pt")
    X_values_val_tokens = tokenizer(X_values_val, padding=True, truncation=True, return_tensors="pt")

    X_headers_train_tokens=tokenizer(X_headers_train, padding=True, truncation=True, return_tensors="pt")
    X_headers_test_tokens=tokenizer(X_headers_test, padding=True, truncation=True, return_tensors="pt")
    X_headers_val_tokens=tokenizer(X_headers_val, padding=True, truncation=True, return_tensors="pt")
    

    with torch.no_grad():
        X_values_train_embeddings = model(**X_values_train_tokens).last_hidden_state.mean(dim=1)
        X_values_test_embeddings = model(**X_values_test_tokens).last_hidden_state.mean(dim=1)
        X_values_val_embeddings = model(**X_values_val_tokens).last_hidden_state.mean(dim=1)
        X_headers_train_embeddings=model(**X_headers_train_tokens).last_hidden_state.mean(dim=1)
        X_headers_test_embeddings=model(**X_headers_test_tokens).last_hidden_state.mean(dim=1)
        X_headers_val_embeddings=model(**X_headers_val_tokens).last_hidden_state.mean(dim=1)

        
    """

    label_encoder = LabelEncoder()
    combined_labels = np.concatenate([y_train, y_test, y_val])
    label_encoder.fit(combined_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    label_encoder_path = "label_encoder_fairtracks.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    return (
        X_values_train_embeddings,
        X_values_test_embeddings,
        X_values_val_embeddings,
        X_headers_train_embeddings,
        X_headers_test_embeddings,
        X_headers_val_embeddings,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
        label_encoder,
        combined_labels,
    )


def to_tensor(
    X_values_train_embeddings,
    X_values_test_embeddings,
    X_values_val_embeddings,
    X_headers_train_embeddings,
    X_headers_test_embeddings,
    X_headers_val_embeddings,
    y_train_encoded,
    y_test_encoded,
    y_val_encoded,
):

    X_values_train_tensor = torch.tensor(X_values_train_embeddings, dtype=torch.float32)
    X_values_test_tensor = torch.tensor(X_values_test_embeddings, dtype=torch.float32)
    X_values_val_tensor = torch.tensor(X_values_val_embeddings, dtype=torch.float32)

    X_headers_train_tensor = torch.tensor(
        X_headers_train_embeddings, dtype=torch.float32
    )
    X_headers_test_tensor = torch.tensor(X_headers_test_embeddings, dtype=torch.float32)
    X_headers_val_tensor = torch.tensor(X_headers_val_embeddings, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    return (
        X_values_train_tensor,
        X_values_test_tensor,
        X_values_val_tensor,
        X_headers_train_tensor,
        X_headers_test_tensor,
        X_headers_val_tensor,
        y_train_tensor,
        y_test_tensor,
        y_val_tensor,
    )


def model_training_validation(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, output_size
):
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss_train = 0.0
        total_correct_train = 0
        total_samples_train = 0
        for values_batch, headers_batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(values_batch, headers_batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            _, predicted_train = torch.max(outputs, 1)
            correct_train = (predicted_train == labels).sum().item()
            total_correct_train += correct_train
            total_samples_train += labels.size(0)
        train_accuracy = total_correct_train / total_samples_train
        train_accuracies.append(train_accuracy)
        train_loss = total_loss_train / len(X_values_train_tensor)
        train_losses.append(train_loss)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Training Accuracy:{train_accuracy}"
        )

        model.eval()
        total_loss_val = 0.0
        total_correct_val = 0
        total_samples_val = 0
        with torch.no_grad():
            y_score = model(X_values_val_tensor, X_headers_val_tensor).cpu().numpy()
            for values_batch, headers_batch, labels in val_loader:
                outputs = model(values_batch, headers_batch)
                loss = loss_fn(outputs, labels)
                total_loss_val = loss.item()
                _, predicted_val = torch.max(outputs, 1)
                correct_val = (predicted_val == labels).sum().item()
                total_correct_val += correct_val
                total_samples_val += labels.size(0)
        validation_accuracy = total_correct_val / total_samples_val
        val_accuracies.append(validation_accuracy)
        val_loss = total_loss_val / len(X_values_val_tensor)
        val_losses.append(val_loss)
    final_train_accuracy = train_accuracies[-1]
    final_val_accuracy = val_accuracies[-1]
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    logger.info(
        f"Final Training Accuracy:{final_train_accuracy}, Final Training Loss:{final_train_loss}"
    )
    logger.info(
        f"Final Validation Accuracy:{final_val_accuracy}, Final Validation Loss:{final_val_loss}"
    )
    # calculating roc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(output_size):
        fpr[i], tpr[i], _ = roc_curve((y_val_tensor == i).cpu(), y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    model_path = "trained_model_fairtracks.pth"
    torch.save(model, model_path)
    return train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc


def model_testing(model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    model.eval()
    total_loss_test = 0.0
    total_correct_test = 0
    total_samples_test = 0
    with torch.no_grad():
        for values_batch, headers_batch, labels in test_loader:
            outputs = model(values_batch, headers_batch)
            loss = loss_fn(outputs, labels)
            total_loss_test += loss.item()
            _, predicted_test = torch.max(outputs, 1)
            correct_test = (predicted_test == labels).sum().item()
            total_correct_test += correct_test
            total_samples_test += labels.size(0)
            all_preds.extend(predicted_test.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_accuracy = total_correct_test / total_samples_test
    test_loss = total_loss_test / len(test_loader)
    logger.info(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
    return all_preds, all_labels, outputs


def json_output(
    all_labels, all_preds, label_encoder, outputs, num_categories, output_file_path
):
    deencoded_labels = label_encoder.inverse_transform(all_labels)
    deencoded_preds = label_encoder.inverse_transform(all_preds)

    grouped_preds = [
        deencoded_preds[i : i + num_categories]
        for i in range(0, len(all_preds), num_categories)
    ]
    grouped_labels = [
        deencoded_labels[i : i + num_categories]
        for i in range(0, len(all_labels), num_categories)
    ]
    consensus = []
    labels = []
    top_three_preds = []
    for i, category_preds in enumerate(grouped_preds):
        counts = Counter(category_preds)
        total_predictions = len(category_preds)
        consensus_value = max(counts, key=counts.get)
        consensus_percentage = counts[consensus_value] / total_predictions
        top_three = {
            pred: count / total_predictions
            for pred, count in counts.items()
            if count / total_predictions > 0.55
        }
        top_three_preds.append(top_three)
        consensus.append((consensus_value, consensus_percentage))
    for i, category_labels in enumerate(grouped_labels):
        counts = Counter(category_labels)
        most_popular_label = max(counts, key=counts.get)
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


def confusion_mat(all_labels, all_preds, combined_labels):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(combined_labels),
        yticklabels=np.unique(combined_labels),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix_model5.jpg")
    plt.show()
    # Calculate class-wise accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    # Print class-wise accuracy
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {i}: {acc:.4f}")


def learning_curve(
    num_epochs, train_accuracies, val_accuracies, train_losses, val_losses
):
    # accuracy
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_model5.jpg")
    plt.show()
    # loss
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_model5.jpg")
    plt.show()


def auc_roc_curve(fpr, tpr, roc_auc, output_size):
    plt.figure(figsize=(12, 12))
    for i in range(output_size):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label="ROC curve (class %d) (AUC = %0.2f)" % (i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_model5.jpg")
    plt.show()


if __name__ == "__main__":
    values_file_path = "../data/encode_subset_metadata.csv"
    headers_file_path = "../data/encode_subset_metadata.csv"
    # load the data into values and headers
    df_values = load(values_file_path)
    df_headers = load(headers_file_path)
    # train, test_validation split
    (
        X_values_train,
        X_values_test,
        X_values_val,
        X_headers_train,
        X_headers_test,
        X_headers_val,
        y_train,
        y_test,
        y_val,
    ) = data_split(df_values, df_headers)
    (
        X_values_train_embeddings,
        X_values_test_embeddings,
        X_values_val_embeddings,
        X_headers_train_embeddings,
        X_headers_test_embeddings,
        X_headers_val_embeddings,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
        label_encoder,
        combined_labels,
    ) = encoding(
        X_values_train,
        X_values_test,
        X_values_val,
        X_headers_train,
        X_headers_test,
        X_headers_val,
        y_train,
        y_test,
        y_val,
    )
    (
        X_values_train_tensor,
        X_values_test_tensor,
        X_values_val_tensor,
        X_headers_train_tensor,
        X_headers_test_tensor,
        X_headers_val_tensor,
        y_train_tensor,
        y_test_tensor,
        y_val_tensor,
    ) = to_tensor(
        X_values_train_embeddings,
        X_values_test_embeddings,
        X_values_val_embeddings,
        X_headers_train_embeddings,
        X_headers_test_embeddings,
        X_headers_val_embeddings,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
    )
    # defining model parameters
    input_size_values = X_values_train_tensor.shape[1]
    input_size_headers = X_headers_train_tensor.shape[1]
    hidden_size = 256
    output_size = len(label_encoder.classes_)
    # Instantiating the model
    model = sentence_transformer_NN(
        input_size_values,
        input_size_headers,
        hidden_size,
        output_size,
        dropout_prob=0.5,
    )
    loss_fn = nn.CrossEntropyLoss()
    # Adding L2 regularization
    l2_reg_lambda = 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg_lambda)

    train_data = TensorDataset(
        X_values_train_tensor, X_headers_train_tensor, y_train_tensor
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = TensorDataset(X_values_val_tensor, X_headers_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_data, batch_size=32)

    test_data = TensorDataset(
        X_values_test_tensor, X_headers_test_tensor, y_test_tensor
    )
    test_loader = DataLoader(test_data, batch_size=32)
    # training and testing the model
    num_epochs = 20
    train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc = (
        model_training_validation(
            model, optimizer, loss_fn, train_loader, val_loader, num_epochs, output_size
        )
    )

    all_preds, all_labels, outputs = model_testing(model, test_loader, loss_fn)
    # json output
    num_categories = len(label_encoder.classes_)
    output_file_path = "predictions_model5.json"
    json_output(
        all_labels, all_preds, label_encoder, outputs, num_categories, output_file_path
    )
    # calculating preciison, recall, f1
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    logger.info(f"Precision:{precision}, Recall: {recall}, F1 Score: {f1}")
    # generating confusion matrix
    confusion_mat(all_labels, all_preds, combined_labels)
    # generating learning curves - accuracy and loss
    learning_curve(
        num_epochs, train_accuracies, val_accuracies, train_losses, val_losses
    )
    # generating AUC ROC
    auc_roc_curve(fpr, tpr, roc_auc, output_size)
