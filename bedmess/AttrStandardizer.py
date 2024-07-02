import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pephubclient import PEPHubClient
from .utils import data_preprocessing, data_encoding
from .model import BoWSTModel
from huggingface_hub import hf_hub_download

# logging set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_from_pephub(pep):
    """
    Fetches metadata from PEPhub registry.
    csv_file (str): Path to the PEPhub registry containing the metadata csv file
    Returns:
        csv_file_path (str): path to the CSV file on the local system.
    """
    phc = PEPHubClient()
    project = phc.load_project(pep)
    sample_table = project.sample_table
    csv_file_df = pd.DataFrame(sample_table)
    return csv_file_df


def load_from_huggingface(schema):
    """
    Load a model from HuggingFace based on the schema of choice.
    Args:
        schema (str): Schema Type
    Returns:
        model : Loaded model
    """
    if schema == "ENCODE":
        # TODO : Change this
        model = hf_hub_download(
            repo_id="databio/attribute-standardizer-model6", filename="model_encode.pth"
        )
    elif schema == "FAIRTRACKS":
        model = None
    return model


def standardize_attr_names(csv_file, schema):
    """
    Standardize attribute names.
    Args:
        csv_file (str): Path to the CSV file containing metadata to be standardized.
        schema (str): Schema type.
    Returns:
        dict: Suggestions for standardized attribute names.
    """
    # X_values_st_tensor, X_values_bow_tensor, X_headers_st_tensor = data_preprocessing(csv_file)
    X_values_st, X_headers_st, X_values_bow = data_preprocessing(csv_file)
    (
        X_headers_embeddings_tensor,
        X_values_embeddings_tensor,
        X_values_bow_tensor,
        label_encoder,
    ) = data_encoding(X_values_st, X_headers_st, X_values_bow, schema)
    logger.info("Data Preprocessing completed.")

    model = load_from_huggingface(schema)
    print(model)
    state_dict = torch.load(model)
    # Padding the input tensors
    # TODO remove the intermediary target_size_* variables, directly declare the padded variables.
    target_size_values = state_dict["fc_values1.weight"].shape[1]
    target_size_headers = state_dict["fc_headers1.weight"].shape[1]
    target_size_values_embeddings = state_dict["fc_values_embeddings1.weight"].shape[1]

    padded_data_values_tensor = torch.zeros(
        X_values_bow_tensor.shape[0], target_size_values
    )
    padded_data_headers_tensor = torch.zeros(
        X_headers_embeddings_tensor.shape[0], target_size_headers
    )
    padded_data_values_embeddings_tensor = torch.zeros(
        X_values_embeddings_tensor.shape[0], target_size_values_embeddings
    )

    padded_data_values_tensor[:, : X_values_bow_tensor.shape[1]] = X_values_bow_tensor
    padded_data_headers_tensor[:, : X_headers_embeddings_tensor.shape[1]] = (
        X_headers_embeddings_tensor
    )
    padded_data_values_embeddings_tensor[:, : X_values_embeddings_tensor.shape[1]] = (
        X_values_embeddings_tensor
    )

    # TODO Should the initialization be mentioned elsewhere?

    input_size_values = padded_data_values_tensor.shape[1]
    input_size_headers = padded_data_headers_tensor.shape[1]
    input_size_values_embeddings = padded_data_values_embeddings_tensor.shape[1]
    hidden_size = 64
    output_size = len(label_encoder.classes_)
    dropout_prob = 0.13
    model = BoWSTModel(
        input_size_values,
        input_size_values_embeddings,
        input_size_headers,
        hidden_size,
        output_size,
        dropout_prob,
    )

    model.load_state_dict(state_dict)

    # Prediction
    # TODO should this be another function?
    model.eval()

    all_preds = []
    all_confidences = []
    with torch.no_grad():
        outputs = model(
            padded_data_values_tensor,
            padded_data_values_embeddings_tensor,
            padded_data_headers_tensor,
        )
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        all_preds.extend(predicted.tolist())
        all_confidences.extend(confidence.tolist())

    decoded_predictions = label_encoder.inverse_transform(all_preds)
    num_categories = len(X_headers_st)
    num_predictions = len(decoded_predictions)

    suggestions = {}
    for i, category in enumerate(X_headers_st):
        if all_confidences[i] >= 0.51:
            prediction = decoded_predictions[i]
            probability = all_confidences[i]
        else:
            prediction = "Not Predictable"
            probability = 0.0
        suggestions[category] = {prediction: probability}

    return suggestions


def AttrStandardizer(pep, schema):
    csv_file = fetch_from_pephub(pep)
    suggestions = standardize_attr_names(csv_file, schema)
    print(suggestions)
