"""
This module has the class AttrStandardizer for 'bedms'.
"""

import logging
import glob
import os
import yaml
from typing import Dict, Tuple, Union, Optional
import pickle
import peppy
import torch
from torch import nn
import torch.nn.functional as torch_functional
import yaml
from huggingface_hub import hf_hub_download
from .const import (
    AVAILABLE_SCHEMAS,
    CONFIDENCE_THRESHOLD,
    PROJECT_NAME,
    SENTENCE_TRANSFORMER_MODEL,
)
from .model import BoWSTModel
from .utils import data_encoding, data_preprocessing, fetch_from_pephub, get_any_pep


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(PROJECT_NAME)


class AttrStandardizer:
    """
    This is the AttrStandardizer class which holds the models for Attribute Standardization.
    """

    def __init__(
        self,
        repo_id: str,
        model_name: str,
        custom_param: Optional[str] = None,
        confidence: int = CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initializes the attribute standardizer with user provided schema, loads the model.

        :param str repo_id: HuggingFace repository ID
        :param str model_name: Name of the schema model
        :param str custom_param: User provided config file for
            custom parameters, if they choose "CUSTOM" schema.
        :param int confidence: Confidence threshold for the predictions.
        """
        self.repo_id = repo_id
        self.model_name = model_name
        self.conf_threshold = confidence
        self.custom_param = custom_param
        self.model, self.vectorizer, self.label_encoder = self._load_model()

    def _get_parameters(self) -> Tuple[int, int, int, int, int, float]:
        """
        Get the model parameters as per the chosen schema.

        :return Tuple[int, int, int, int, int, int, float]: Tuple containing the model parameters.
        """
        config_filename = f"config_{self.model_name}.yaml"
        config_pth = hf_hub_download(
            repo_id=self.repo_id,
            filename=os.path.join(self.model_name, config_filename),
        )
        with open(config_pth, "r") as file:
            config = yaml.safe_load(file)

        input_size_bow = config["params"]["input_size_bow"]
        embedding_size = config["params"]["embedding_size"]
        hidden_size = config["params"]["hidden_size"]
        output_size = config["params"]["output_size"]
        dropout_prob = config["params"]["dropout_prob"]

        return (
            input_size_bow,
            embedding_size,
            embedding_size,
            hidden_size,
            output_size,
            dropout_prob,
        )

    def _load_model(self) -> Tuple[nn.Module, object, object]:
        """
        Calls function to load the model from HuggingFace repository
          load vectorizer and label encoder and sets to eval().
        :return nn.Module: Loaded Neural Network Model.
        :return object: The scikit learn vectorizer for bag of words encoding.
        :return object: Label encoder object for the labels (y).
        """
        model_filename = f"model_{self.model_name}.pth"
        label_encoder_filename = f"label_encoder_{self.model_name}.pkl"
        vectorizer_filename = f"vectorizer_{self.model_name}.pkl"

        model_pth = hf_hub_download(
            repo_id=self.repo_id, filename=os.path.join(self.model_name, model_filename)
        )

        vc_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=os.path.join(self.model_name, vectorizer_filename),
        )

        lb_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=os.path.join(self.model_name, label_encoder_filename),
        )

        with open(vc_path, "rb") as f:
            vectorizer = pickle.load(f)

        with open(lb_path, "rb") as f:
            label_encoder = pickle.load(f)

        state_dict = torch.load(model_pth)

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
