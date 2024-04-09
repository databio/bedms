import nn_model1_preprocess 
from nn_model1_model import NN1
from nn_model1_train import ModelTraining

import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

def main(input_file_path):
    try:
        encoder_path="nn_model1_encoder.pth"
        combined_labels, X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor = nn_model1_preprocess.preprocessing(input_file_path, encoder_path)
        logger.info("Data Preprocessing Completed.")

        input_size=X_train_tensor.shape[1]
        hidden_size=64
        output_size=len(np.unique(combined_labels))

        model=NN1(input_size, hidden_size,output_size)
        loss_fn=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.05)
        trainer=ModelTraining(model, loss_fn, optimizer)
        train_accuracies, val_accuracies, train_losses, val_losses=trainer.train(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=10, batch_size=32, device='cpu')
        logger.info("Model Training Completed.")

        model_path="nn1_model.pth"
        trainer.save_model(model_path)

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")

if __name__=="__main__":
    input_file_path="../data/dummy_1.tsv"
    main(input_file_path)