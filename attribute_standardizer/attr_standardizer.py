import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
import logging
import peppy

from typing import Dict, Tuple, Union

from .model import BoWSTModel
from .utils import (
    fetch_from_pephub,
    load_from_huggingface,
    data_preprocessing,
    data_encoding,
    get_any_pep,
)
from .const import (
    HIDDEN_SIZE,
    DROPOUT_PROB,
    CONFIDENCE_THRESHOLD,
    EMBEDDING_SIZE,
    SENTENCE_TRANSFORMER_MODEL,
    INPUT_SIZE_BOW_FAIRTRACKS,
    INPUT_SIZE_BOW_ENCODE,
    OUTPUT_SIZE_ENCODE,
    OUTPUT_SIZE_FAIRTRACKS,
    INPUT_SIZE_BOW_BEDBASE,
    OUTPUT_SIZE_BEDBASE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttrStandardizer:
    def __init__(self, schema: str, confidence: int = CONFIDENCE_THRESHOLD) -> None:
        """
        Initializes the attribute standardizer with user provided schema, loads the model.

        :param str schema: User provided schema, can be "ENCODE" or "FAIRTRACKS"
        :param int confidence: Confidence threshold for the predictions.
        """
        self.schema = schema
        self.model = self._load_model()
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
        elif self.schema == "FAIRTRACKS":
            return (
                INPUT_SIZE_BOW_FAIRTRACKS,
                EMBEDDING_SIZE,
                EMBEDDING_SIZE,
                HIDDEN_SIZE,
                OUTPUT_SIZE_FAIRTRACKS,
                DROPOUT_PROB,
            )
        elif self.schema == "BEDBASE":
            return (
                INPUT_SIZE_BOW_BEDBASE,
                EMBEDDING_SIZE,
                EMBEDDING_SIZE,
                HIDDEN_SIZE,
                OUTPUT_SIZE_BEDBASE,
                DROPOUT_PROB,
            )
        else:
            raise ValueError(
                f"Schema not available: {self.schema}. Presently, three schemas are available: ENCODE , FAIRTRACKS, BEDBASE"
            )

    def _load_model(self) -> nn.Module:
        """
        Calls function to load the model from HuggingFace repository and sets to eval().

        :return nn.Module: Loaded Neural Network Model.
        """
        try:
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
            return model

        except Exception as e:
            logger.error(f"Error loading the model: {str(e)}")
            raise

    def standardize(
        self, pep: Union[str, peppy.Project]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetches the user provided PEP from the PEPHub registry path, returns the predictions.

        :param str pep: peppy.Project object or PEPHub registry path to PEP.
        :return Dict[str, Dict[str, float]]: Suggestions to the user.
        """
        if isinstance(pep, str):
            pep = get_any_pep(pep)
        elif isinstance(pep, peppy.Project):
            pass
        else:
            raise ValueError(
                f"PEP should be either a path to PEPHub registry or peppy.Project object."
            )
        try:
            csv_file = fetch_from_pephub(pep)
            schema = self.schema
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

            with torch.no_grad():
                outputs = self.model(
                    X_values_bow_tensor,
                    X_values_embeddings_tensor,
                    X_headers_embeddings_tensor,
                )
                probabilities = torch_functional.softmax(outputs, dim=1)
                # confidence, predicted = torch.max(probabilities, 1)

                values, indices = torch.topk(probabilities, k=3, dim=1)
                top_preds = indices.tolist()
                top_confidences = values.tolist()

                decoded_predictions = [
                    label_encoder.inverse_transform(indices) for indices in top_preds
                ]

                suggestions = {}
            for i, category in enumerate(X_headers_st):
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
