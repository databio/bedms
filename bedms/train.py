""" This is the training script with which the user can train their own models."""

import logging
import torch
from torch import nn
from torch import optim
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import yaml
from .utils_train import (
    load_from_dir,
    accumulate_data,
    training_encoding,
    data_loader,
    train_model,
    plot_learning_curve,
    model_testing,
    plot_confusion_matrix,
    auc_roc_curve,
)
from .const import PROJECT_NAME, EMBEDDING_SIZE
from .model import BoWSTModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(PROJECT_NAME)


class AttrStandardizerTrainer:
    """
    This is the training class responsible for
    managing the training process for the standardizer model.
    """

    def __init__(self, config: str) -> None:
        """
        Initializes the TrainStandardizer object with the given configuration.

        :param str config: Path to the config file which has the training parameters provided by the user.
        """
        self.label_encoder = None
        self.vectorizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.output_size = None
        self.criterion = None
        self.train_accuracies = None
        self.val_accuracies = None
        self.train_losses = None
        self.val_losses = None
        self.model = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self.all_labels = None
        self.all_preds = None

        with open(config, "r") as file:
            self.config = yaml.safe_load(file)

    def load_encode_data(self) -> None:
        """
        Loads and prepares the encoded training, testing and validation datasets.
        """
        values_files_list = load_from_dir(self.config["dataset"]["values_dir_pth"])
        headers_files_list = load_from_dir(self.config["dataset"]["headers_dir_pth"])

        if len(values_files_list) != len(headers_files_list):
            logger.error(
                f"Mismatch in number of value files ({len(values_files_list)}) \
                and header files ({len(headers_files_list)})"
            )
            return

        total_files = len(values_files_list)

        paired_files = list(zip(values_files_list, headers_files_list))

        train_size = self.config["data_split"]["train_set"]
        test_size = self.config["data_split"]["test_set"]
        val_size = self.config["data_split"]["val_set"]

        if train_size + val_size + test_size > total_files:
            logger.error(
                f"Data split sizes exceed total number of files: "
                f"train({train_size}) + val({val_size}) + \
                test({test_size}) > total_files({total_files})"
            )
            return

        train_files = paired_files[:train_size]
        val_files = paired_files[train_size : train_size + val_size]
        test_files = paired_files[
            train_size + val_size : train_size + val_size + test_size
        ]

        logger.info(f"Training on {len(train_files)} file sets")
        logger.info(f"Validating on {len(val_files)} file sets")
        logger.info(f"Testing on {len(test_files)} file sets")

        x_values_train_list, x_headers_train_list, y_train_list = accumulate_data(
            train_files
        )
        x_values_test_list, x_headers_test_list, y_test_list = accumulate_data(
            test_files
        )
        x_values_val_list, x_headers_val_list, y_val_list = accumulate_data(val_files)

        logger.info("Accumulation Done.")

        num_cluster = self.config["training"]["num_cluster"]
        vectorizer_pth = self.config["training"]["vectorizer_pth"]
        label_encoder_pth = self.config["training"]["label_encoder_pth"]
        sentence_transformer_model = self.config["training"][
            "sentence_transformer_model"
        ]

        (
            train_encoded_data,
            test_encoded_data,
            val_encoded_data,
            self.label_encoder,
            self.vectorizer,
        ) = training_encoding(
            x_values_train_list,
            x_headers_train_list,
            y_train_list,
            x_values_test_list,
            x_headers_test_list,
            y_test_list,
            x_values_val_list,
            x_headers_val_list,
            y_val_list,
            num_cluster,
            vectorizer_pth,
            label_encoder_pth,
            sentence_transformer_model,
        )
        logger.info("Encoding Done.")

        batch_size = self.config["training"]["batch_size"]
        self.train_loader = data_loader(train_encoded_data, batch_size)
        self.test_loader = data_loader(test_encoded_data, batch_size)
        self.val_loader = data_loader(val_encoded_data, batch_size)

        logger.info("Loading Done.")

    def training(self):
        """
        Trains the model.
        """
        input_size_values = len(self.vectorizer.vocabulary_)
        input_size_values_embeddings = EMBEDDING_SIZE
        input_size_headers = EMBEDDING_SIZE
        hidden_size = self.config["model"]["hidden_size"]
        self.output_size = len(self.label_encoder.classes_)  # Number of classes
        dropout_prob = self.config["model"]["dropout_prob"]

        self.model = BoWSTModel(
            input_size_values,
            input_size_values_embeddings,
            input_size_headers,
            hidden_size,
            self.output_size,
            dropout_prob,
        )

        learning_rate = self.config["training"]["learning_rate"]
        self.criterion = nn.CrossEntropyLoss()
        l2_reg_lambda = self.config["training"]["l2_regularization"]
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=l2_reg_lambda
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Training the model
        num_epochs = self.config["training"]["num_epochs"]

        model_pth = self.config["training"]["model_pth"]
        bow_drops = self.config["training"]["bow_drops"]

        (
            self.train_accuracies,
            self.val_accuracies,
            self.train_losses,
            self.val_losses,
            self.fpr,
            self.tpr,
            self.roc_auc,
        ) = train_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.criterion,
            optimizer,
            device,
            num_epochs,
            self.output_size,
            model_pth,
            bow_drops,
        )

        logger.info("Training Done.")

    def testing(self):
        """
        Model testing.
        """
        self.all_preds, self.all_labels = model_testing(
            self.model, self.test_loader, self.criterion
        )
        precision = precision_score(self.all_labels, self.all_preds, average="macro")
        recall = recall_score(self.all_labels, self.all_preds, average="macro")
        f1 = f1_score(self.all_labels, self.all_preds, average="macro")
        logger.info(f"Precision:{precision}, Recall: {recall}, F1 Score: {f1}")

    def plot_visualizations(self):
        """
        Generates visualizations for training ( accuracy and loss curves)
        and testing( confusion matrix, roc curve)
        """
        num_epochs = self.config["training"]["num_epochs"]
        accuracy_fig_pth = self.config["visualization"]["accuracy_fig_pth"]
        loss_fig_pth = self.config["visualization"]["loss_fig_pth"]
        cm_pth = self.config["visualization"]["confusion_matrix_fig_pth"]
        roc_pth = self.config["visualization"]["roc_fig_pth"]
        plot_learning_curve(
            num_epochs,
            self.train_accuracies,
            self.val_accuracies,
            self.train_losses,
            self.val_losses,
            accuracy_fig_pth,
            loss_fig_pth,
        )
        plot_confusion_matrix(
            self.all_labels, self.all_preds, self.label_encoder.classes_, cm_pth
        )
        auc_roc_curve(self.fpr, self.tpr, self.roc_auc, self.output_size, roc_pth)
