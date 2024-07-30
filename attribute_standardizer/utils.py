import pandas as pd
import numpy as np
import torch
from pephubclient import PEPHubClient
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from huggingface_hub import hf_hub_download
from typing import Optional, Any, List, Tuple, Union
from .const import (
    REPO_ID,
    FILENAME_ENCODE,
    FILENAME_FAIRTRACKS,
    ENCODE_LABEL_ENCODER_FILENAME,
    FAIRTRACKS_LABEL_ENCODER_FILENAME,
    ENCODE_VECTORIZER_FILENAME,
    FAIRTRACKS_VECTORIZER_FILENAME,
    SENTENCE_TRANSFORMER_MODEL,
)


def fetch_from_pephub(pep: str) -> pd.DataFrame:
    """
    Fetches metadata from PEPhub registry.

    :param str pep: Path to the PEPhub registry containing the metadata csv file
    :return pd.DataFrame: path to the CSV file on the local system.
    """
    phc = PEPHubClient()
    project = phc.load_project(pep)
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
        model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_ENCODE)
    elif schema == "FAIRTRACKS":
        model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_FAIRTRACKS)
    return model


def data_preprocessing(
    df: pd.DataFrame,
) -> Tuple[List[List[str]], List[str], List[List[str]]]:
    """
    Preprocessing the DataFrame by extracting the column values and headers.

    :param pd.DataFrame df: The input DataFrame (user chosen PEP) to preprocess.
    :return Tuple[List[List[str]], List[str], List[List[str]]]:
        - Nested list containing the comma separated values in each column for sentence transformer embeddings.
        - List containing the headers of the DataFrame.
        - Nested list containing the comma separated values in each column for Bag of Words encoding.
    """

    X_values_st = [df[column].astype(str).tolist() for column in df.columns]
    X_headers_st = df.columns.tolist()
    X_values_bow = [df[column].astype(str).tolist() for column in df.columns]

    return X_values_st, X_headers_st, X_values_bow


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


def data_encoding(
    X_values_st: List[List[str]],
    X_headers_st: List[str],
    X_values_bow: List[List[str]],
    schema: str,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[LabelEncoder, None]]:
    """
    Encode input data in accordance with the user-specified schemas.

    :param list X_values_st: Nested list containing the comma separated values in each column for sentence transformer embeddings.
    :param list X_headers_st: List containing the headers of the DataFrame.
    :param list X_values_bow: Nested list containing the comma separated values in each column for Bag of Words encoding.
    :param str schema: Schema type chosen by the user for standardization.
    :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[LabelEncoder, None]]: Tuple containing torch tensors for encoded embeddings and Bag of Words representations, and label encoder object.
    """
    # Sentence Transformer Model
    sentence_encoder = SentenceTransformer(model_name)
    X_headers_embeddings = sentence_encoder.encode(
        X_headers_st, show_progress_bar=False
    )
    # generating embeddings for each element in sublist (column)
    embeddings = []
    for column in X_values_st:
        val_embedding = sentence_encoder.encode(column, show_progress_bar=False)
        embedding = get_top_k_average(val_embedding, k=3)
        embeddings.append(embedding)
    X_values_embeddings = embeddings
    if schema == "ENCODE":
        # Bag of Words Vectorizer
        vectorizer = CountVectorizer()
        vc_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=ENCODE_VECTORIZER_FILENAME,
        )
        with open(vc_path, "rb") as f:
            vectorizer = pickle.load(f)
        transformed_columns = []
        for column in X_values_bow:
            column_text = " ".join(column)
            transformed_column = vectorizer.transform([column_text])
            transformed_columns.append(transformed_column.toarray()[0])
        transformed_columns = np.array(transformed_columns)
        # print(transformed_columns)
        X_values_bow = transformed_columns
        # Label Encoding
        label_encoder = LabelEncoder()
        lb_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=ENCODE_LABEL_ENCODER_FILENAME,
        )
        with open(lb_path, "rb") as f:
            label_encoder = pickle.load(f)

    elif schema == "FAIRTRACKS":
        vectorizer = CountVectorizer()
        vc_path = hf_hub_download(
            repo_id=REPO_ID, filename=FAIRTRACKS_VECTORIZER_FILENAME
        )
        with open(vc_path, "rb") as f:
            vectorizer = pickle.load(f)
        transformed_columns = []
        for column in X_values_bow:
            column_text = " ".join(column)
            transformed_column = vectorizer.transform([column_text])
            transformed_columns.append(transformed_column.toarray()[0])
        transformed_columns = np.array(transformed_columns)
        # print(transformed_columns)
        X_values_bow = transformed_columns
        # Label Encoding
        label_encoder = LabelEncoder()
        lb_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FAIRTRACKS_LABEL_ENCODER_FILENAME,
        )
        with open(lb_path, "rb") as f:
            label_encoder = pickle.load(f)

    X_headers_embeddings_tensor = torch.tensor(
        X_headers_embeddings, dtype=torch.float32
    )
    X_values_embeddings_tensor = torch.tensor(X_values_embeddings, dtype=torch.float32)
    X_values_bow_tensor = torch.tensor(X_values_bow, dtype=torch.float32)

    return (
        X_headers_embeddings_tensor,
        X_values_embeddings_tensor,
        X_values_bow_tensor,
        label_encoder,
    )
