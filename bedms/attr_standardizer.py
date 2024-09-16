"""
This module has the class AttrStandardizer for 'bedms'.
"""

import logging
from typing import Dict, Tuple, Union
import pickle
import peppy
import torch
from torch import nn
import torch.nn.functional as torch_functional

from .const import (
    AVAILABLE_SCHEMAS,
    CONFIDENCE_THRESHOLD,
    DROPOUT_PROB,
    EMBEDDING_SIZE,
    HIDDEN_SIZE,
    INPUT_SIZE_BOW_BEDBASE,
    INPUT_SIZE_BOW_ENCODE,
    INPUT_SIZE_BOW_FAIRTRACKS,
    OUTPUT_SIZE_BEDBASE,
    OUTPUT_SIZE_ENCODE,
    OUTPUT_SIZE_FAIRTRACKS,
    PROJECT_NAME,
    SENTENCE_TRANSFORMER_MODEL,
    REPO_ID,
    ENCODE_VECTORIZER_FILENAME,
    ENCODE_LABEL_ENCODER_FILENAME,
    FAIRTRACKS_VECTORIZER_FILENAME,
    FAIRTRACKS_LABEL_ENCODER_FILENAME,
    BEDBASE_VECTORIZER_FILENAME,
    BEDBASE_LABEL_ENCODER_FILENAME,
)
from .model import BoWSTModel
from .utils import (
    data_encoding,
    data_preprocessing,
    fetch_from_pephub,
    get_any_pep,
    load_from_huggingface,
)
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(PROJECT_NAME)


class AttrStandardizer:
    """
    This is the AttrStandardizer class which holds the models for Attribute Standardization.
    """

    def __init__(self, schema: str, confidence: int = CONFIDENCE_THRESHOLD) -> None:
        """
        Initializes the attribute standardizer with user provided schema, loads the model.

        :param str schema: User provided schema, can be "ENCODE" or "FAIRTRACKS"
        :param int confidence: Confidence threshold for the predictions.
        """
        self.schema = schema
        self.model, self.vectorizer, self.label_encoder = self._load_model()
        self.conf_threshold = confidence

    def _get_parameters(self) -> Tuple[int, int, int, int, int, float]:
        """
        Get the model parameters as per the chosen schema.

        :return Tuple[int, int, int, int, int, int, float]: Tuple containing the model parameters.
        """
        if self.schema == "ENCODE":
            return (
                INPUT_SIZE_BOW_ENCODE,
                EMBEDDING_SIZE,
                EMBEDDING_SIZE,
                HIDDEN_SIZE,
                OUTPUT_SIZE_ENCODE,
                DROPOUT_PROB,
            )
        if self.schema == "FAIRTRACKS":
            return (
                INPUT_SIZE_BOW_FAIRTRACKS,
                EMBEDDING_SIZE,
                EMBEDDING_SIZE,
                HIDDEN_SIZE,
                OUTPUT_SIZE_FAIRTRACKS,
                DROPOUT_PROB,
            )
        if self.schema == "BEDBASE":
            return (
                INPUT_SIZE_BOW_BEDBASE,
                EMBEDDING_SIZE,
                EMBEDDING_SIZE,
                HIDDEN_SIZE,
                OUTPUT_SIZE_BEDBASE,
                DROPOUT_PROB,
            )
        raise ValueError(
            f"Schema not available: {self.schema}."
            "Presently, three schemas are available: ENCODE , FAIRTRACKS, BEDBASE"
        )

    def _load_model(self) -> Tuple[nn.Module, object, object]:
        """
        Calls function to load the model from HuggingFace repository
          load vectorizer and label encoder and sets to eval().
        :return nn.Module: Loaded Neural Network Model.
        :return object: The scikit learn vectorizer for bag of words encoding.
        :return object: Label encoder object for the labels (y).
        """
        try:
            if self.schema == "ENCODE":
                filename_vc = ENCODE_VECTORIZER_FILENAME
                filename_lb = ENCODE_LABEL_ENCODER_FILENAME
            elif self.schema == "FAIRTRACKS":
                filename_vc = FAIRTRACKS_VECTORIZER_FILENAME
                filename_lb = FAIRTRACKS_LABEL_ENCODER_FILENAME
            elif self.schema == "BEDBASE":
                filename_vc = BEDBASE_VECTORIZER_FILENAME
                filename_lb = BEDBASE_LABEL_ENCODER_FILENAME

            vectorizer = None
            label_encoder = None

            vc_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename_vc,
            )

            with open(vc_path, "rb") as f:
                vectorizer = pickle.load(f)

            lb_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename_lb,
            )

            with open(lb_path, "rb") as f:
                label_encoder = pickle.load(f)

            model = load_from_huggingface(self.schema)
            state_dict = torch.load(model)

            (
                input_size_values,
                input_size_values_embeddings,
                input_size_headers,
                hidden_size,
                output_size,
                dropout_prob,
            ) = self._get_parameters()

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
            return model, vectorizer, label_encoder

        except Exception as e:
            logger.error(f"Error loading the model: {str(e)}")
            raise

    def standardize(
        self, pep: Union[str, peppy.Project]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetches the user provided PEP
        from the PEPHub registry path,
        returns the predictions.

        :param str pep: peppy.Project object or PEPHub registry path to PEP.
        :return Dict[str, Dict[str, float]]: Suggestions to the user.
        """
        if isinstance(pep, str):
            pep = get_any_pep(pep)
        elif isinstance(pep, peppy.Project):
            pass
        else:
            raise ValueError(
                "PEP should be either a path to PEPHub registry or peppy.Project object."
            )
        try:
            csv_file = fetch_from_pephub(pep)

            x_values_st, x_headers_st, x_values_bow, num_rows = data_preprocessing(
                csv_file
            )
            (
                x_headers_embeddings_tensor,
                x_values_embeddings_tensor,
                x_values_bow_tensor,
                label_encoder,
            ) = data_encoding(
                self.vectorizer,
                self.label_encoder,
                num_rows,
                x_values_st,
                x_headers_st,
                x_values_bow,
                model_name=SENTENCE_TRANSFORMER_MODEL,
            )

            logger.info("Data Preprocessing completed.")

            with torch.no_grad():
                outputs = self.model(
                    x_values_bow_tensor,
                    x_values_embeddings_tensor,
                    x_headers_embeddings_tensor,
                )
                probabilities = torch_functional.softmax(outputs, dim=1)

                values, indices = torch.topk(probabilities, k=3, dim=1)
                top_preds = indices.tolist()
                top_confidences = values.tolist()

                decoded_predictions = [
                    label_encoder.inverse_transform(indices) for indices in top_preds
                ]

                suggestions = {}
            for i, category in enumerate(x_headers_st):
                category_suggestions = {}
                if top_confidences[i][0] >= self.conf_threshold:
                    for j in range(3):
                        prediction = decoded_predictions[i][j]
                        probability = top_confidences[i][j]
                        if probability >= self.conf_threshold:
                            category_suggestions[prediction] = probability
                        else:
                            break
                else:
                    category_suggestions["Not Predictable"] = 0.0

                suggestions[category] = category_suggestions

            return suggestions

        except Exception as e:
            logger.error(
                f"Error occured during standardization in standardize function: {str(e)}"
            )

    @staticmethod
    def get_available_schemas() -> list[str]:
        """
        Stores a list of available schemas.

        :return list: List of available schemas.
        """

        return AVAILABLE_SCHEMAS
