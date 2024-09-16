"""
This module has all util functions for 'bedms'
"""

import warnings
from collections import Counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import peppy
import torch
from huggingface_hub import hf_hub_download

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from .const import (
    MODEL_BEDBASE,
    MODEL_ENCODE,
    MODEL_FAIRTRACKS,
    NUM_CLUSTERS,
    REPO_ID,
    PEP_FILE_TYPES,
)

# TODO : convert to single np array before converting to tensor
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow.",
)


def fetch_from_pephub(project: peppy.Project) -> pd.DataFrame:
    """
    Fetches metadata from PEPhub registry.

    :param str pep: Path to the PEPhub registry containing the metadata csv file
    :return pd.DataFrame: path to the CSV file on the local system.
    """

    sample_table = project.sample_table
    csv_file_df = pd.DataFrame(sample_table)
    return csv_file_df


def load_from_huggingface(schema: str) -> Optional[Any]:
    """
    Load a model from HuggingFace based on the schema of choice.

    :param str schema: Schema Type
    :return Optional[Any]: Loaded model object
    """
    if schema == "ENCODE":
        model = hf_hub_download(repo_id=REPO_ID, filename=MODEL_ENCODE)
    elif schema == "FAIRTRACKS":
        model = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FAIRTRACKS)
    elif schema == "BEDBASE":
        model = hf_hub_download(repo_id=REPO_ID, filename=MODEL_BEDBASE)
    return model


def data_preprocessing(
    df: pd.DataFrame,
) -> Tuple[List[List[str]], List[str], List[List[str]], int]:
    """
    Preprocessing the DataFrame by extracting the column values and headers.

    :param pd.DataFrame df: The input DataFrame (user chosen PEP) to preprocess.
    :return Tuple[List[List[str]], List[str], List[List[str]]]:
        - Nested list containing the comma separated values
        in each column for sentence transformer embeddings.
        - List containing the headers of the DataFrame.
        - Nested list containing the comma separated values
        in each column for Bag of Words encoding.
        - Number of rows in the metadata csv
    """

    x_values_st = [df[column].astype(str).tolist() for column in df.columns]
    x_headers_st = df.columns.tolist()
    x_values_bow = [df[column].astype(str).tolist() for column in df.columns]

    num_rows = df.shape[0]

    return x_values_st, x_headers_st, x_values_bow, num_rows


def get_top_k_average(val_embedding: List[np.ndarray], k: int) -> np.ndarray:
    """
    Calculates the average of the top k most common embeddings.

    :param list val_embedding: List of embeddings, each embedding is a vector of values.
    :param int k: The number of top common embeddings to consider.
    :return np.ndarray: The mean of the top k most common embeddings as a NumPy array.
    """

    embeddings_list = [tuple(embedding) for embedding in val_embedding]
    counts = Counter(embeddings_list)
    top_3_embeddings = [
        np.array(embedding) for embedding, count in counts.most_common(k)
    ]
    top_3_embeddings_tensor = torch.tensor(top_3_embeddings)
    column_embedding_mean = torch.mean(top_3_embeddings_tensor, dim=0)

    return column_embedding_mean.numpy()


def get_top_cluster_averaged(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the average of the largest embedding cluster.

    :param list embeddings: List of embeddings, each embedding is a vector of values.
    :return np.ndarray: The mean of the largest cluster as a NumPy array.
    """
    flattened_embeddings = [embedding.tolist() for embedding in embeddings]
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(flattened_embeddings)
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

    return top_k_average.numpy()


def get_averaged(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Averages the embeddings.
    :param list embeddings: List of embeddings, each embedding is a vector of values.
    :return np.ndarray: The mean of all the embeddings as a NumPy array.
    """
    flattened_embeddings = [embedding.tolist() for embedding in embeddings]
    flattened_embeddings_array = np.array(flattened_embeddings)
    averaged_embedding = np.mean(flattened_embeddings_array, axis=0)

    return averaged_embedding


def data_encoding(
    vectorizer: object,
    label_encoder: object,
    num_rows: int,
    x_values_st: List[List[str]],
    x_headers_st: List[str],
    x_values_bow: List[List[str]],
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[LabelEncoder, None]]:
    """
    Encode input data in accordance with the user-specified schemas.

    :param object vectorizer: scikit-learn vectorizer for bag of words encoding.
    :param object label_encoder" Label encoder object storing labels (y)
    :param int num_rows: Number of rows in the sample metadata
    :param list X_values_st: Nested list containing the comma separated values
    in each column for sentence transformer embeddings.
    :param list X_headers_st: List containing the headers of the DataFrame.
    :param list X_values_bow: Nested list containing the comma separated values
    in each column for Bag of Words encoding.
    :param str schema: Schema type chosen by the user for standardization.
    :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    Union[LabelEncoder, None]]: Tuple containing
    torch tensors for encoded embeddings and Bag of Words representations,
    and label encoder object.
    """
    # Sentence Transformer Model
    sentence_encoder = SentenceTransformer(model_name)
    x_headers_embeddings = sentence_encoder.encode(
        x_headers_st, show_progress_bar=False
    )
    # generating embeddings for each element in sublist (column)
    embeddings = []
    for column in x_values_st:
        val_embedding = sentence_encoder.encode(column, show_progress_bar=False)
        if num_rows >= 10:
            embedding = get_top_cluster_averaged(val_embedding)
        else:
            embedding = get_averaged(val_embedding)

        embeddings.append(embedding)
    x_values_embeddings = embeddings
    transformed_columns = []
    for column in x_values_bow:
        column_text = " ".join(column)
        transformed_column = vectorizer.transform([column_text])
        transformed_columns.append(transformed_column.toarray()[0])
    transformed_columns = np.array(transformed_columns)
    # print(transformed_columns)
    x_values_bow = transformed_columns

    x_headers_embeddings_tensor = torch.tensor(
        x_headers_embeddings, dtype=torch.float32
    )
    x_values_embeddings_tensor = torch.tensor(x_values_embeddings, dtype=torch.float32)
    x_values_bow_tensor = torch.tensor(x_values_bow, dtype=torch.float32)
    x_values_embeddings_tensor = x_values_embeddings_tensor.squeeze(
        1
    )  # brings the shape to [num_cols, vocab]

    return (
        x_headers_embeddings_tensor,
        x_values_embeddings_tensor,
        x_values_bow_tensor,
        label_encoder,
    )


def get_any_pep(pep: str) -> peppy.Project:
    """
    Get the PEP file from the local system or from PEPhub.

    :param pep: Path to the PEP file or PEPhub registry path.

    :return: peppy.Project object.
    """
    res = list(filter(pep.endswith, PEP_FILE_TYPES)) != []
    if res:
        return peppy.Project(pep)
    return peppy.Project.from_pephub(pep)
