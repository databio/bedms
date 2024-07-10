import torch
import torch.nn as nn
import torch.nn.functional as F


##NN model - Bag of Words + Sentence Transformer Model
class BoWSTModel(nn.Module):
    """Neural Network model combining Bag of Words and Sentence Transformer embeddings."""

    def __init__(
        self,
        input_size_values: int,
        input_size_values_embeddings: int,
        input_size_headers: int,
        hidden_size: int,
        output_size: int,
        dropout_prob: float,
    ) -> None:
        """
        Initializes the BoWSTModel.

        :param int input_size_values: Size of the input for the values (BoW).
        :param int inout_size_values_embeddings: Size of the input for the values sentence transformer embeddings.
        :param int input_size_headers: Size of the input for the headers with sentence transformer embeddings.
        :param int hidden_size: Size of the hidden layer.
        :param int output_size: Size of the output layer.
        :param float dropout_prob: Dropout probability for regularization.
        """
        super(BoWSTModel, self).__init__()
        self.fc_values1 = nn.Linear(input_size_values, hidden_size)
        self.dropout_values1 = nn.Dropout(dropout_prob)
        self.fc_values2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_values2 = nn.Dropout(dropout_prob)
        self.fc_values_embeddings1 = nn.Linear(
            input_size_values_embeddings, hidden_size
        )
        self.dropout_values_embeddings1 = nn.Dropout(dropout_prob)
        self.fc_values_embeddings2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_values_embeddings2 = nn.Dropout(dropout_prob)
        self.fc_headers1 = nn.Linear(input_size_headers, hidden_size)
        self.dropout_headers1 = nn.Dropout(dropout_prob)
        self.fc_headers2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_headers2 = nn.Dropout(dropout_prob)
        self.fc_combined1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout_combined1 = nn.Dropout(dropout_prob)
        self.fc_combined2 = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x_values: torch.Tensor,
        x_values_embeddings: torch.Tensor,
        x_headers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the model.

        :param torch.Tensor x_values: Input tensor for the values (BoW)
        :param torch.Tensor x_values_embeddings: Input tensor for the value embeddings.
        :param torch.Tensor x_headers: Input tensor for the headers.
        :return torch.Tensor: Output tesnor after passing through the model.
        """
        x_values = F.relu(self.fc_values1(x_values))
        x_values = self.dropout_values1(x_values)
        x_values = F.relu(self.fc_values2(x_values))
        x_values = self.dropout_values2(x_values)
        x_values_embeddings = F.relu(self.fc_values_embeddings1(x_values_embeddings))
        x_values_embeddings = self.dropout_values_embeddings1(x_values_embeddings)
        x_values_embeddings = F.relu(self.fc_values_embeddings2(x_values_embeddings))
        x_values_embeddings = self.dropout_values_embeddings2(x_values_embeddings)
        x_headers = F.relu(self.fc_headers1(x_headers))
        x_headers = self.dropout_headers1(x_headers)
        x_headers = F.relu(self.fc_headers2(x_headers))
        x_headers = self.dropout_headers2(x_headers)

        x_combined = torch.cat((x_values, x_values_embeddings, x_headers), dim=1)
        x_combined = F.relu(self.fc_combined1(x_combined))
        x_combined = self.dropout_combined1(x_combined)
        x_combined = self.fc_combined2(x_combined)
        return x_combined
