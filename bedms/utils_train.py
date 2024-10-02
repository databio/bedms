"""
This module has all training util functions for 'bedms'
"""

import os
import logging
from glob import glob
import warnings
from collections import Counter
from typing import List, Tuple, Iterator, Dict
import pickle
import random


import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from .const import PROJECT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(PROJECT_NAME)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow.",
)


def load_from_dir(dir: str) -> List[str]:
    """
    Loads each file from the directory path.

    :param str dir: Path to the directory.
    :return: List:paths to each file in the directory.
    """
    return glob(os.path.join(dir, "*.csv"))


def load_and_preprocess(file_path: str) -> pd.DataFrame:
    """
    Loads and Preprocesses each csv file as a Pandas DataFrame.

    :param str file_path: Path to each csv file.
    :return pandas.DataFrame: df of each csv file.
    """
    df = pd.read_csv(file_path, sep=",")
    df.replace("NA", np.nan, inplace=True)
    for column in df.columns:
        most_common_val = df[column].mode().iloc[0]
        df[column] = df[column].fillna(most_common_val)
    return df


def accumulate_data(
    files: List[Tuple[str, str]]
) -> Tuple[List[List[List[str]]], List[List[List[str]]], List[pd.Index]]:
    """
    Accumulates data from multiple files into lists.

    :param List[Tuple[str, str]] files: List containing
        sublists of values or header files.
    :return Tuple[List[List[List[str]]], List[List[List[str]]],[List[str]]:
        Lists of values, headers, labels.
    A tuple containing three lists:
        - A nested list of values (list of tables where
            each table is a list of lists for columns),
        - A nested list of headers (similar structure to values),
        - A list of Pandas Index objects containing column labels.
    """
    x_values_list = []
    x_headers_list = []
    y_list = []
    for values_file, headers_file in files:
        df_values = load_and_preprocess(values_file)
        df_headers = load_and_preprocess(headers_file)
        df_values = df_values.fillna("")
        df_headers = df_headers.fillna("")
        y = df_values.columns
        table_list = []
        # values list
        for col in df_values.columns:
            sublist_list = df_values[col].tolist()
            table_list.append(sublist_list)
        x_values_list.append(table_list)
        # headers list
        table_list = []
        for col in df_headers.columns:
            sublist_list = df_headers[col].tolist()
            table_list.append(sublist_list)
        x_headers_list.append(table_list)
        # y list
        y_list.append(y)

    return x_values_list, x_headers_list, y_list


def lazy_loading(data_list: List, batch_size: int) -> Iterator[List]:
    """
    Lazy loading for data in batches.

    :param List data_list: List of data to be loaded lazily.
    :param int batch_size: Size of batch.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i : i + batch_size]


def get_top_training_cluster_averaged(
    embeddings: List[torch.tensor], num: int
) -> torch.Tensor:
    """
    Computes the clutser-averaged top training embeddings using k-means clustering.

    :param List[torch.tensor] embeddings: List of embedding tensors to cluster.
    :param int num: Number of clusters to be created using k-means.
    :return torch.Tensor: A tensor representing the
        average of embeddings in the most common cluster.
    """
    flattened_embeddings = [embedding.tolist() for embedding in embeddings]
    kmeans = KMeans(n_clusters=num, random_state=0).fit(flattened_embeddings)
    labels_kmeans = kmeans.labels_
    cluster_counts = Counter(labels_kmeans)
    most_common_cluster = max(cluster_counts, key=cluster_counts.get)
    most_common_indices = [
        idx for idx, label in enumerate(labels_kmeans) if label == most_common_cluster
    ]
    most_common_embeddings = [
        torch.tensor(embeddings[idx]) for idx in most_common_indices
    ]

    if most_common_embeddings:
        top_k_average = torch.mean(
            torch.stack(most_common_embeddings), dim=0
        ).unsqueeze(0)
    else:
        top_k_average = torch.zeros_like(most_common_embeddings[0]).unsqueeze(0)
    return top_k_average


def training_encoding(
    x_values_train_list: List[List[List[str]]],
    x_headers_train_list: List[List[List[str]]],
    y_train_list: List[pd.Index],
    x_values_test_list: List[List[List[str]]],
    x_headers_test_list: List[List[List[str]]],
    y_test_list: List[pd.Index],
    x_values_val_list: List[List[List[str]]],
    x_headers_val_list: List[List[List[str]]],
    y_val_list: List[pd.Index],
    num_cluster: int,
    vectorizer_pth: str,
    label_encoder_pth: str,
    sentence_transformer_model: str,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    LabelEncoder,
    List[str],
    CountVectorizer,
]:
    """
    Generates encoded headers and values.

    :param List[List[List[str]]] x_values_train_list:
        Nested list containing the training set for values.
    :param List[List[List[str]]] x_headers_train_list:
        Nested list containing the training set for headers.
    :param List[pd.Index] y_train_list:
        List of the column labels ( attributes) for training.
    :param List[List[List[str]]] x_values_test_list:
        Nested list containing the testing set for values.
    :param List[List[List[str]]] x_headers_test_list:
        Nested list containing the testing set for headers.
    :param List[pd.Index] y_test_list:
        List of the column labels ( attributes) for testing.
    :param List[List[List[str]]] x_values_val_list:
        Nested list containing the validation set for values.
    :param List[List[List[str]]] x_headers_val_list:
        Nested list containing the validation set for headers.
    :param List[pd.Index] y_val_list:
        List of the column labels ( attributes) for validation.
    :return Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    LabelEncoder,
    List[str],
    CountVectorizer]: Returns a tuple of
     - training dataset tensor
     - testing dataset tensor
     - validation dataset tensor
     - trained label encoder
     - list of unique values encountered during training
     - Trained vectorizer for Bag of Words representation

    """
    # Bag of Words
    flattened_list = [
        item for sublist in x_values_train_list for col in sublist for item in col
    ]
    vectorizer = CountVectorizer()
    vectorizer.fit(flattened_list)
    with open(vectorizer_pth, "wb") as f:
        pickle.dump(vectorizer, f)
    vocabulary_size = len(vectorizer.vocabulary_)
    logger.info(f"Vocabulary size: {vocabulary_size}")

    # Sentence Transformers
    model_name = sentence_transformer_model
    sentence_encoder = SentenceTransformer(model_name)

    # Label Encoders
    label_encoder = LabelEncoder()
    flat_y_train = [",".join(y) for y in y_train_list]
    individual_values = [value.strip() for y in flat_y_train for value in y.split(",")]
    unique_values = set(individual_values)
    unique_values_list = list(unique_values)
    label_encoder.fit(unique_values_list)

    with open(label_encoder_pth, "wb") as f:
        pickle.dump(label_encoder, f)

    def encode_data(
        x_values_list: List[List[List[str]]],
        x_headers_list: List[List[List[str]]],
        y_list: List[pd.Index],
        num_cluster: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This nested function encodes the values, headers and labels data.
        It is called for thrice - training, testing, validation.

        :param List[List[List[str]]] x_values_list: Nested list containing values.
        :param List[List[List[str]]] x_headers_list: Nested list containing headers.
        :param List[pd.Index] y_list: Labels (attributes) list.
        :param int num_cluster: Number of clusters to be generated.
        """
        x_values_bow_tensors = []
        x_values_embeddings_tensors = []
        x_headers_embeddings_tensors = []
        y_tensors = []

        for x_values, x_headers, y in zip(x_values_list, x_headers_list, y_list):

            for i in range(len(x_values)):  # Iterate over columns
                # BoW Representation
                x_values_bow = vectorizer.transform(x_values[i]).toarray()
                x_values_bow_tensor = (
                    torch.tensor(x_values_bow, dtype=torch.float32)
                    .mean(dim=0)
                    .unsqueeze(0)
                    .clone()
                    .detach()
                )

                # Embeddings for Values
                embeddings_values = [
                    sentence_encoder.encode(str(value), show_progress_bar=False)
                    for value in x_values[i]
                ]

                top_k_average_values = get_top_training_cluster_averaged(
                    embeddings_values, num_cluster
                )  # Average of all embeddings
                x_values_embeddings_tensor = top_k_average_values.clone().detach()

                # Embeddings for Headers
                embeddings_headers = [
                    sentence_encoder.encode(str(header), show_progress_bar=False)
                    for header in x_headers[i]
                ]

                top_k_average_headers = get_top_training_cluster_averaged(
                    embeddings_headers, num_cluster
                )  # Average of all embeddings
                x_headers_embeddings_tensor = top_k_average_headers.clone().detach()

                # Labels
                y_col = label_encoder.transform([y[i]])
                y_col_tensor = torch.tensor(y_col, dtype=torch.long).clone().detach()

                x_values_bow_tensors.append(x_values_bow_tensor)
                x_values_embeddings_tensors.append(x_values_embeddings_tensor)
                x_headers_embeddings_tensors.append(x_headers_embeddings_tensor)
                y_tensors.append(y_col_tensor)

        x_values_bow_tensor = torch.cat(
            x_values_bow_tensors, dim=0
        )  # this has [num_cols, vocab_size]
        x_values_embeddings_tensor = torch.cat(
            x_values_embeddings_tensors, dim=0
        )  # [num_cols, embedding_dim]
        x_headers_embeddings_tensor = torch.cat(x_headers_embeddings_tensors, dim=0)
        y_tensor = torch.cat(y_tensors, dim=0)  # [num_cols]

        return (
            x_values_bow_tensor,
            x_values_embeddings_tensor,
            x_headers_embeddings_tensor,
            y_tensor,
        )

    train_data = encode_data(
        x_values_train_list, x_headers_train_list, y_train_list, num_cluster
    )
    test_data = encode_data(
        x_values_test_list, x_headers_test_list, y_test_list, num_cluster
    )
    val_data = encode_data(
        x_values_val_list, x_headers_val_list, y_val_list, num_cluster
    )

    return (
        train_data,
        test_data,
        val_data,
        label_encoder,
        vectorizer,
    )


def data_loader(
    encoded_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> DataLoader:
    """
    Creates a DataLoader from encoded tensor data.

    :param [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] encoded_data:
        Tuple containing tensors for
    values bag of words, values embeddings, headers embeddings, and labels.
    :param int batch_size: The number of samples per batch for the DataLoader.
    :return DataLoader: A PyTorch DataLoader which yields
        batches of data from the given tensors.
    """
    (
        x_values_bow_tensor,
        x_values_embeddings_tensor,
        x_headers_embeddings_tensor,
        y_tensor,
    ) = encoded_data
    # Convert data to TensorDataset
    dataset = TensorDataset(
        x_values_bow_tensor,
        x_values_embeddings_tensor,
        x_headers_embeddings_tensor,
        y_tensor,
    )
    # Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def drop_bow(bow_tensor: torch.Tensor, num_drops: int) -> torch.Tensor:
    """
    Randomly drops a specified number of columns in the
    Bag of Words tensor for regularization.

    :param torch.Tensor bow_tensor: Bag of Words tensor.
    :param int num_drops: Number of columns to be randomly
        dropped from the Bag of Words tensor.
    :return torch.Tensor: Bag of Words tensor with dropped columns.
    """
    num_columns = bow_tensor.size(0)
    columns = list(range(num_columns))
    columns_to_drop = random.sample(columns, num_drops)

    mask = torch.ones(num_columns, dtype=torch.bool)
    mask[columns_to_drop] = False
    mask = mask.unsqueeze(1).expand_as(bow_tensor)

    # Apply the mask to the BoW tensor
    dropped_bow_tensor = bow_tensor.clone()
    dropped_bow_tensor[~mask] = 0.0

    return dropped_bow_tensor


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    output_size: int,
    model_pth: str,
    bow_drops: int,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, float],
]:
    """
    Trains and validates the neural network model.

    :param torch.nn.Module model: The neural network model to be trained.
    :param DataLoader train_loader: DataLoader for the training set.
    :param DataLoader val_loader: DataLoader for the validation set.
    :param torch.nn.Module criterion: The loss function used to compute loss during training.
    :param torch.optim.Optimizer optimizer: The optimizer to update the model parameters.
    :param torch.device device: The device (CPU or GPU) on which the model will be trained.
    :param int num_epochs: The number of epochs to train the model.
    :param int output_size: The size of the model's output layer.
    :param str model_pth: The file path to where the model would be saved.
    :param int bow_drops: The number of Bag of Words columns to be dropped.
    :return Tuple:
     - List[float]: Train accuracy per epoch.
     - List[float]: Validation accuracy per epoch.
     - List[float]: Train loss per epoch.
     - List[float]: Validation loss per epoch.
     - Dict[int, np.ndarray]: Dictionary of False Positive Rates (FPR).
     - Dict[int, np.ndarray]: Dictionary of True Positive Rates (TPR).
     - Dict[int, float]: Dictionary of Area Under the ROC Curve for different classes.
    """
    patience = 3
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop = False

    model.train()

    for epoch in range(num_epochs):
        total_samples = 0
        correct_predictions = 0
        train_loss = 0.0
        for x_values_bow, x_values_embeddings, x_headers_embeddings, y in train_loader:
            x_values_bow = x_values_bow.to(device)
            x_values_embeddings = x_values_embeddings.to(device)
            x_headers_embeddings = x_headers_embeddings.to(device)
            y = y.to(device)

            x_values_bow = drop_bow(x_values_bow, bow_drops)

            optimizer.zero_grad()
            outputs = model(x_values_bow, x_values_embeddings, x_headers_embeddings)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_values_bow.size(0)

            _, predicted = torch.max(outputs, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()

        train_accuracy = correct_predictions / total_samples * 100
        train_accuracies.append(train_accuracy)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct_predictions_val = 0
        total_samples_val = 0
        y_true = []
        y_scores = []
        with torch.no_grad():
            for (
                x_values_bow,
                x_values_embeddings,
                x_headers_embeddings,
                y,
            ) in val_loader:
                x_values_bow = x_values_bow.to(device)
                x_values_embeddings = x_values_embeddings.to(device)
                x_headers_embeddings = x_headers_embeddings.to(device)
                y = y.to(device)
                outputs = model(x_values_bow, x_values_embeddings, x_headers_embeddings)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x_values_bow.size(0)

                _, predicted = torch.max(outputs, 1)
                total_samples_val += y.size(0)
                correct_predictions_val += (predicted == y).sum().item()
                y_true.extend(y.cpu().numpy())
                y_scores.extend(outputs.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_predictions_val / total_samples_val * 100
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, \
            Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, \
            Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Early stop

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_pth)
        elif epoch - best_epoch >= patience:
            early_stop = True
    if early_stop:
        print(f"Early stop at {best_epoch + 1} epoch.")
    y_true = label_binarize(y_true, classes=list(range(output_size)))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Calculate ROC curves and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(output_size):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return train_accuracies, val_accuracies, train_losses, val_losses, fpr, tpr, roc_auc


def model_testing(
    model: torch.nn.Module, test_loader: DataLoader, loss_fn: torch.nn.Module
) -> Tuple[List[int], List[int], torch.Tensor]:
    """
    This functions tests the model.

    :param torch.nn.Module model: The trained model.
    :param DataLoader test_loader: DataLoader for the testing set.
    :param torch.nn.Module loss_fn: The loss function used to compute loss.
    :return Tuple:
        - List[int]: List of all the predictions made by the model.
        - List[int]: List of all the true labels ( Ground truth)
        - torch.Tensor: Logist from the model for the test dataset.
    """
    all_preds = []
    all_labels = []
    model.eval()
    total_loss_test = 0.0
    total_correct_test = 0
    total_samples_test = 0
    with torch.no_grad():
        for values_batch, bow_batch, headers_batch, labels in test_loader:
            outputs = model(values_batch, bow_batch, headers_batch)
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

    return all_preds, all_labels


def plot_learning_curve(
    num_epochs: int,
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_losses: List[float],
    val_losses: List[float],
    accuracy_fig_pth: str,
    loss_fig_pth: str,
) -> None:
    """
    Plots the learning curves - accuracy and loss for Training and Validation of the model.

    :param int num_epochs: Number of epochs for which the model was trained.
    :param List[float] train_accuracies: List of training accuracies for each epoch.
    :param List[float] val_accuracies: List of validation accuracies for each epoch.
    :param List[float] train_losses: List of training losses for each epoch.
    :param List[float] val_losses: List of validation losses for each epoch.
    :param str accuracy_fig_pth: Path where the accuracy curve figure will be saved.
    :param str loss_fig_pth: Path where the loss curve figure will be saved.
    """

    # accuracy
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_fig_pth, format="svg")
    plt.show()
    plt.close()
    # loss
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_fig_pth, format="svg")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    unique_values_list: List[str],
    confusion_matrix_fig_pth: str,
) -> None:
    """
    Plots confusion matrix for the test data.

    :param List[int] y_true: List of true labels ( Ground Truth)
    :param List[int] y_pred: List of predictions made by the model.
    :param List[str] unique_values_list: List of all the classes that the model predicted.
    :param str confusion_matrix_fig_pth: Path where the confusion matrix figure will be saved.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(unique_values_list),
        yticklabels=np.unique(unique_values_list),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(confusion_matrix_fig_pth, format="svg")
    plt.show()
    plt.close()
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {i}: {acc:.4f}")


def auc_roc_curve(
    fpr: Dict[int, np.ndarray],
    tpr: Dict[int, np.ndarray],
    roc_auc: Dict[int, float],
    output_size: int,
    roc_fig_pth: str,
) -> None:
    """
    Plots the ROC Curve.

    :param Dict[int, np.ndarray] fpr: Dictionary of False Positive Rates
    :param Dicr[int, np.ndarray] tpr: Dictionary of True Positive Rates
    :param Dict[int, float] roc_auc: Dictionary of Area Under Curve for ROC for different classes.
    :param int output_size: The number of classes the model predicted into.
    :param str roc_fig_pth: Path to where the ROC figure will be saved.
    """
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
    plt.savefig(roc_fig_pth, format="svg")
    plt.show()
    plt.close()
