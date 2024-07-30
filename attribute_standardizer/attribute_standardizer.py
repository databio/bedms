import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .const import (
    HIDDEN_SIZE,
    DROPOUT_PROB,
    CONFIDENCE_THRESHOLD,
    SENTENCE_TRANSFORMER_MODEL,
)

from .utils import (
    fetch_from_pephub,
    load_from_huggingface,
    data_preprocessing,
    data_encoding,
)
from .model import BoWSTModel
from huggingface_hub import hf_hub_download
from typing import Dict, List, Tuple, Any, Union


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_attr_names(csv_file: str, schema: str) -> Dict[str, Dict[str, float]]:
    """
    Standardize attribute names.

    :param str csv_file: Path to the CSV file containing metadata to be standardized.
    :param str schema: Schema type.
    :return Dict[str, Dict[str, float]]: Suggestions for standardized attribute names.
    """
    try:
        X_values_st, X_headers_st, X_values_bow = data_preprocessing(csv_file)
        (
            X_headers_embeddings_tensor,
            X_values_embeddings_tensor,
            X_values_bow_tensor,
            label_encoder,
        ) = data_encoding(
            X_values_st,
            X_headers_st,
            X_values_bow,
            schema,
            model_name=SENTENCE_TRANSFORMER_MODEL,
        )
        logger.info("Data Preprocessing completed.")

        model = load_from_huggingface(schema)
        # print(model)
        state_dict = torch.load(model)

        """Padding the input tensors."""

        padded_data_values_tensor = torch.zeros(
            X_values_bow_tensor.shape[0], state_dict["fc_values1.weight"].shape[1]
        )
        padded_data_headers_tensor = torch.zeros(
            X_headers_embeddings_tensor.shape[0],
            state_dict["fc_headers1.weight"].shape[1],
        )
        padded_data_values_embeddings_tensor = torch.zeros(
            X_values_embeddings_tensor.shape[0],
            state_dict["fc_values_embeddings1.weight"].shape[1],
        )

        padded_data_values_tensor[:, : X_values_bow_tensor.shape[1]] = (
            X_values_bow_tensor
        )
        padded_data_headers_tensor[:, : X_headers_embeddings_tensor.shape[1]] = (
            X_headers_embeddings_tensor
        )
        padded_data_values_embeddings_tensor[
            :, : X_values_embeddings_tensor.shape[1]
        ] = X_values_embeddings_tensor

        input_size_values = padded_data_values_tensor.shape[1]
        input_size_headers = padded_data_headers_tensor.shape[1]
        input_size_values_embeddings = padded_data_values_embeddings_tensor.shape[1]
        hidden_size = HIDDEN_SIZE
        output_size = len(label_encoder.classes_)
        dropout_prob = DROPOUT_PROB
        model = BoWSTModel(
            input_size_values,
            input_size_values_embeddings,
            input_size_headers,
            hidden_size,
            output_size,
            dropout_prob,
        )

        model.load_state_dict(state_dict)

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

        suggestions = {}
        for i, category in enumerate(X_headers_st):
            if all_confidences[i] >= CONFIDENCE_THRESHOLD:
                prediction = decoded_predictions[i]
                probability = all_confidences[i]
            else:
                prediction = "Not Predictable"
                probability = 0.0
            suggestions[category] = {prediction: probability}

        return suggestions
    except Exception as e:
        logger.error(f"Error occured in standardize_attr_names: {str(e)}")
        return {}


def attr_standardizer(pep: str, schema: str) -> None:
    """
    :param str pep: Path to the PEPhub registry containing the metadata csv file.
    :param str schema: Schema Type chosen by the user.
    """
    if not pep:
        raise ValueError(
            "pep argument is missing or empty. Please provide the PEPHub registry path to PEP"
        )
    if not schema:
        raise ValueError(
            "schema argument is missing or empty. Please mention the schema of choice: ENCODE or FAIRTRACKS."
        )
    csv_file = fetch_from_pephub(pep)
    suggestions = standardize_attr_names(csv_file, schema)

    logger.info(suggestions)
