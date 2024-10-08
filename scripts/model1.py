""" Training & Model for Model Architecture 1 ( One Hot Encoding of Values Only)"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NN1(nn.Module):
    """Simple Neural Network with a single Hidden Layer."""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the NN1 model.

        :param int input_size: The number of input features.
        :param int hidden_size: The size of the hidden layer.
        :param int output_size: The size of the output.
        """
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: Output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load(file_path):
    """
    Loads a CSV file into a DataFrame. Imputes missing values with most common value in that column.

    :param str file_path: The path to the CSV file to be loaded.
    :return pd.DataFrame: Imputed DataFrame.
    """
    df = pd.read_csv(file_path, sep=",")
    df.replace("NA", np.nan, inplace=True)

    for column in df.columns:
        most_common_val = df[column].mode().iloc[0]
        df[column] = df[column].fillna(most_common_val)
    return df


def data_split(df_values):
    """
    Splits the DataFrame into training, test and validation.

    :param pd.DataFrame df_values: Input DataFrame with values to be split.
    :return list: Features and labels split as train, test, and val (validation).
    """
    df_values_train, df_values_temp = train_test_split(
        df_values, test_size=0.2, random_state=42
    )
    df_values_test, df_values_val = train_test_split(
        df_values_temp, test_size=0.5, random_state=42
    )

    # Snippet for testing on unseen data
    """
    df_values_test = pd.read_csv(
        "/home/saanika/curation/scripts/bedmess_archive/data/encode_metadata_values_moderate.csv",
        sep=",",
    )
    """
    # Comment out the above for training on seen data.

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

    y_train = df_values_train.columns
    y_test = df_values_test.columns
    y_val = df_values_val.columns

    # reshaping
    X_values_train_flat = [item for sublist in X_values_train for item in sublist]
    X_values_test_flat = [item for sublist in X_values_test for item in sublist]
    X_values_val_flat = [item for sublist in X_values_val for item in sublist]

    X_values_train_2d = [[val] for val in X_values_train_flat]
    X_values_test_2d = [[val] for val in X_values_test_flat]
    X_values_val_2d = [[val] for val in X_values_val_flat]

    y_train_expanded = [label for label in y_train for _ in range(len(df_values_train))]
    y_test_expanded = [label for label in y_test for _ in range(len(df_values_test))]
    y_val_expanded = [label for label in y_val for _ in range(len(df_values_val))]

    X_values_train = X_values_train_2d
    X_values_test = X_values_test_2d
    X_values_val = X_values_val_2d
    y_train = y_train_expanded
    y_test = y_test_expanded
    y_val = y_val_expanded

    return X_values_train, X_values_test, X_values_val, y_train, y_test, y_val


def encoding(X_values_train, X_values_test, X_values_val, y_train, y_test, y_val):
    """
    Encodes the values for the model.

    :param list X_values_train: Training features.
    :param list X_values_test: Testing features.
    :param list X_values_val: Validation features.
    :param list y_train: Training labels.
    :param list y_test: Test labels.
    :param list y_val: Valiation labels.
    :param return tuple
    """
    enc_values = OneHotEncoder(handle_unknown="ignore")
    X_values_train_encoded = enc_values.fit_transform(X_values_train)
    X_values_test_encoded = enc_values.transform(X_values_test)
    X_values_val_encoded = enc_values.transform(X_values_val)

    combined_labels = np.concatenate([y_train, y_test, y_val])
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    return (
        X_values_train_encoded,
        X_values_test_encoded,
        X_values_val_encoded,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
        label_encoder,
        combined_labels,
    )


def to_tensor(
    X_values_train_encoded,
    X_values_test_encoded,
    X_values_val_encoded,
    y_train_encoded,
    y_test_encoded,
    y_val_encoded,
):
    """
    Converts the encoded features and labels to tensor.

    :param array X_values_train_encoded: Training features as array.
    :param array X_values_test_encoded: Test features as array.
    :param array X_values_val_encoded: Validation features as array.
    :param array y_train_encoded:  Training labels as an array.
    :param array y_test_encoded: Testing labels as an array.
    :param array y_val_encoded: Validation labels as an array.
    :return tuple: Tensors for training, testing, validation for features and labels.
    """

    X_values_train_tensor = torch.tensor(
        X_values_train_encoded.toarray(), dtype=torch.float32
    )
    X_values_test_tensor = torch.tensor(
        X_values_test_encoded.toarray(), dtype=torch.float32
    )
    X_values_val_tensor = torch.tensor(
        X_values_val_encoded.toarray(), dtype=torch.float32
    )

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    return (
        X_values_train_tensor,
        X_values_test_tensor,
        X_values_val_tensor,
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
        for values_batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(values_batch)
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
            y_score = model(X_values_val_tensor).cpu().numpy()
            for values_batch, labels in val_loader:
                outputs = model(values_batch)
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
    return train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc


def model_testing(model, test_loader, loss_fn):
    start_time = time.time()
    all_preds = []
    all_labels = []
    model.eval()
    total_loss_test = 0.0
    total_correct_test = 0
    total_samples_test = 0
    with torch.no_grad():
        for values_batch, labels in test_loader:
            outputs = model(values_batch)
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
    end_time = time.time()
    test_time = end_time - start_time
    print(f"Total time taken for training:{test_time:.2f} seconds")
    return all_preds, all_labels


def json_output(
    all_preds, all_labels, label_encoder, num_categories, output_file_path
):  # TODO
    decoded_preds = label_encoder.inverse_transform(all_preds)
    decoded_labels = label_encoder.inverse_transform(all_labels)
    grouped_preds = [
        decoded_preds[i : i + num_categories]
        for i in range(0, len(all_preds), num_categories)
    ]
    grouped_labels = [
        decoded_labels[i : i + num_categories]
        for i in range(0, len(all_labels), num_categories)
    ]
    consensus = []
    top_three_preds = []
    labels = []

    for i, category_preds in enumerate(grouped_preds):
        counts = Counter(category_preds)
        total_predictions = len(category_preds)
        consensus_value = max(counts, key=counts.get)
        consensus_percentage = counts[consensus_value] / total_predictions
        top_three_preds.append(
            {pred: count / total_predictions for pred, count in counts.most_common(3)}
        )
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
    plt.savefig("confusion_matrix_model2.jpg")
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
    plt.savefig("accuracy_model2.jpg")
    plt.show()
    # loss
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_model2.jpg")
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
    plt.savefig("roc_curve_model2.jpg")
    plt.show()


if __name__ == "__main__":
    values_file_path = "/home/saanika/curation/scripts/bedmess_archive/data/encode_subset_new_metadata.csv"
    # load the data into values and headers
    df_values = load(values_file_path)
    # train, test, validation split
    X_values_train, X_values_test, X_values_val, y_train, y_test, y_val = data_split(
        df_values
    )
    # One Hot Encoding and Label Encoding
    (
        X_values_train_encoded,
        X_values_test_encoded,
        X_values_val_encoded,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
        label_encoder,
        combined_labels,
    ) = encoding(X_values_train, X_values_test, X_values_val, y_train, y_test, y_val)
    # convert to tensors
    (
        X_values_train_tensor,
        X_values_test_tensor,
        X_values_val_tensor,
        y_train_tensor,
        y_test_tensor,
        y_val_tensor,
    ) = to_tensor(
        X_values_train_encoded,
        X_values_test_encoded,
        X_values_val_encoded,
        y_train_encoded,
        y_test_encoded,
        y_val_encoded,
    )
    # defining the model parameters
    input_size_values = X_values_train_tensor.shape[1]
    hidden_size = 32
    output_size = len(label_encoder.classes_)
    # Instantiating the model
    model = NN1(input_size_values, hidden_size, output_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = TensorDataset(X_values_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = TensorDataset(X_values_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_data, batch_size=32)

    test_data = TensorDataset(X_values_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=32)
    # training and validation of the model
    num_epochs = 20
    train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc = (
        model_training_validation(
            model, optimizer, loss_fn, train_loader, val_loader, num_epochs, output_size
        )
    )
    # testing the model
    all_preds, all_labels = model_testing(model, test_loader, loss_fn)
    # predictions output
    num_categories = len(label_encoder.classes_)
    output_file_path = "predictions_model1.json"
    json_output(all_preds, all_labels, label_encoder, num_categories, output_file_path)
    # calculating precision, recall, f1
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    logger.info(f"Precision:{precision}, Recall: {recall}, F1 Score: {f1}")
    # generatung confusion matrix
    confusion_mat(all_labels, all_preds, combined_labels)
    # generating learning curves - accuracy and loss
    learning_curve(
        num_epochs, train_accuracies, val_accuracies, train_losses, val_losses
    )
    # generating AUC ROC
    auc_roc_curve(fpr, tpr, roc_auc, output_size)
