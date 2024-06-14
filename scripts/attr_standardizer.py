import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
#from transformers import AutoTokenizer, AutoModel
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import logging 
import argparse 
from collections import Counter 
import os
import subprocess
from pephubclient import PEPHubClient

#logging set up
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


class NN(nn.Module):
    """
    Neural Network class for Attribute Standardization.
    Args:
    input_size_values(int): Input size of the values data frame.
    input_size_headers(int): Input size of the headers data frame.
    hidden_size (int): Size of the hidden layer.
    output_size (int) : Output Size.
    dropout_prob(float): Dropout Probability. 

    """
    def __init__(self, input_size_values, input_size_headers, hidden_size, output_size, dropout_prob):
        super(NN, self).__init__()
        self.fc_values1 = nn.Linear(input_size_values, hidden_size)
        self.dropout_values1 = nn.Dropout(dropout_prob)
        self.fc_values2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_values2 = nn.Dropout(dropout_prob)
        self.fc_headers1 = nn.Linear(input_size_headers, hidden_size)
        self.dropout_headers1 = nn.Dropout(dropout_prob)
        self.fc_headers2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_headers2 = nn.Dropout(dropout_prob)
        self.fc_combined1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout_combined1 = nn.Dropout(dropout_prob)
        self.fc_combined2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_values, x_headers):
        x_values = F.relu(self.fc_values1(x_values))
        x_values = self.dropout_values1(x_values)
        x_values = F.relu(self.fc_values2(x_values))
        x_values = self.dropout_values2(x_values)
        x_headers = F.relu(self.fc_headers1(x_headers))
        x_headers = self.dropout_headers1(x_headers)
        x_headers = F.relu(self.fc_headers2(x_headers))
        x_headers = self.dropout_headers2(x_headers)
        x_combined = torch.cat((x_values, x_headers), dim=1)
        x_combined = F.relu(self.fc_combined1(x_combined))
        x_combined = self.dropout_combined1(x_combined)
        x_combined = self.fc_combined2(x_combined)
        return x_combined

class AttrStandardizer():
      # TODO def save_model for uploading model to HuggingFace
      # TODO def upload_to_huggingface 
      """
      Class for Attribute Stndardisation - Training and Prediction.
      Attributes:
          schema : Schema the model will standardize into.
          model : Neural Network Model.
      """
      def __init__(self, model):
            self.schema=None
            self.model=model

      def model_training(self, model, optimizer, loss_fn, train_loader, num_epochs, output_size):
           """
           Training the model.
           Args:
               model (NN): Neural network model.
               optimizer (torch.optim.Optimizer): Optimizer.
               loss_fn: Loss function.
               train_loader (DataLoader): Training data loader.
               num_epochs (int): Number of epochs for training.
               output_size (int): Output size.

           Returns:
               float: Training accuracy.

           """
           self.model.train()
           total_loss_train=0.0
           correct_predictions=0
           total_predictions=0
           for epoch in range(num_epochs):
                for values_batch, headers_batch, labels in train_loader:
                     optimizer.zero_grad()
                     outputs=self.model(values_batch, headers_batch)
                     loss=loss_fn(outputs, labels)
                     loss.backward()
                     optimizer.step()

                     total_loss_train+=loss.item()
                     _, predicted = torch.max(outputs, 1)
                     correct_predictions += (predicted == labels).sum().item()
                     total_predictions += labels.size(0)
                train_accuracy=correct_predictions/total_predictions               

                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss_train/ len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}')
           return train_accuracy
      
      def model_prediction(self, eval_loader):
           """
           Make predictions using the trained model.

           Args:
               eval_loader (DataLoader): Evaluation/User data loader.

           Returns:
               list: Predicted labels.
           """
           self.model.eval()
           predictions=[]
           with torch.no_grad():
                for values_batch, headers_batch in eval_loader:
                     outputs=self.model(values_batch, headers_batch)
                     _, predicted = torch.max(outputs, 1)
                     predictions.extend(predicted.tolist())
           return predictions

def standardize_attr_names_from_path(csv_file):
     """
     Fetches metadata from PEPhub registry.
     csv_file (str): Path to the PEPhub registry containing the metadata csv file
     Returns:
          csv_file_path (str): path to the CSV file on the local system.
     """
     phc=PEPHubClient()
     project = phc.load_project(csv_file)
     sample_table=project.sample_table
     csv_file_df = pd.DataFrame(sample_table)
     return csv_file_df
     
def standardize_attr_names(csv_file:str, schema:str, values_file_path, headers_file_path, flag):
      """

      Standardize attribute names.

      Args:
          csv_file (str): Path to the CSV file containing metadata to be standardized.
          schema (str): Schema type.
          values_file_path (str): Path to the values file.
          headers_file_path (str): Path to the headers file.

      Returns:
          dict: Suggestions for standardized attribute names.
      """
     
      X_values_train_tensor, X_headers_train_tensor, y_train_tensor, label_encoder ,X_values_eval_tensor, X_headers_eval_tensor, X_headers_eval, X_values_eval= \
          data_preprocessing(values_file_path, headers_file_path, csv_file, flag)
      logger.info("Data Preprocessing completed.")
      input_size_values=X_values_train_tensor.shape[1]
      input_size_headers=X_headers_train_tensor.shape[1]
      hidden_size=256
      output_size=len(label_encoder.classes_)
      model=NN(input_size_values, input_size_headers, hidden_size, output_size, dropout_prob=0.5)
      loss_fn=nn.CrossEntropyLoss()
      l2_reg_lambda = 0.001
      optimizer=optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg_lambda)

      train_data=TensorDataset(X_values_train_tensor, X_headers_train_tensor, y_train_tensor)
      train_loader=DataLoader(train_data, batch_size=32, shuffle=True)

      num_epochs=20
      attr_standardizer=AttrStandardizer(model)
      train_accuracy= attr_standardizer.model_training(model, optimizer, loss_fn, train_loader, num_epochs, output_size)
      logger.info("Training Completed.")
      logger.info(f"Final Training Accuracy: {train_accuracy}")
      eval_data=TensorDataset(X_values_eval_tensor, X_headers_eval_tensor)
      eval_loader=DataLoader(eval_data, batch_size=32, shuffle=False)
      predictions=attr_standardizer.model_prediction(eval_loader)
      #de encoding the predictions
      decoded_predictions = label_encoder.inverse_transform(predictions)
      num_categories=len(X_headers_eval)
      num_predictions=len(decoded_predictions)
      predictions_per_category = num_predictions // num_categories
      grouped_preds = [decoded_predictions[i*predictions_per_category : (i+1)*predictions_per_category] for i in range(num_categories)]
     
      #taking consensus
      suggestions={}
      for header, preds in zip(X_headers_eval, grouped_preds):
          pred_counts=Counter(preds)
          total_predictions=len(preds)
          probabilities = {attr: count / total_predictions for attr, count in pred_counts.items() if count/total_predictions>0.5}
          suggestions[header]=probabilities
      return suggestions

def load_data(file_path, flag):
     """
     Load data from a file.

     Args:
          file_path (str): Path to the file.

     Returns:
          pandas.DataFrame: Loaded data.
     """
     if flag=="path":
          df=pd.read_csv(file_path, sep=",", dtype=str)
     elif flag=="df":
          df=file_path
     df.replace('NA', np.nan, inplace=True)
     for column in df.columns:
          if df[column].notna().any(): #skipping those columns that are all empty 
               most_common_val = df[column].mode().iloc[0]
               #print(most_common_val)
               df[column] = df[column].fillna(most_common_val)
     return df 

def data_preprocessing(values_file_path, headers_file_path, csv_file, flag):
     """
     Preprocess the data for training and evaluation.

     Args:
        values_file_path (str): Path to the values file.
        headers_file_path (str): Path to the headers file.
        csv_file (str): Path to the TSV file containing user provided metadata.

     Returns:
        tuple: Processed data tensors
        pandas.DataFrame: User provided metadata .
     """
     df_values=load_data(values_file_path, flag="path")
     df_headers=load_data(headers_file_path, flag="path")
     df_user =load_data(csv_file, flag)

     X_values_train=[df_values[column].astype(str).tolist() for column in df_values.columns]
     X_headers_train=[df_headers[column].astype(str).tolist() for column in df_headers.columns]
     y_train=df_headers.columns
     X_values_eval=[df_user[column].astype(str).tolist() for column in df_user.columns]
     X_headers_eval=df_user.columns

     X_values_train_flat = [item for sublist in X_values_train for item in sublist]
     X_headers_train_flat = [item for sublist in X_headers_train for item in sublist]
     y_train_expanded = [label for label in y_train for _ in range(len(df_values))]
     X_values_eval_flat=[item for sublist in X_values_eval for item in sublist]
     X_headers_eval_expanded=[x for x in X_headers_eval for _ in range(len(df_user))]

     #encoding using Sentence Transformer 
     model_name='all-MiniLM-L6-v2' 
     sentence_encoder = SentenceTransformer(model_name)
     X_values_train_embeddings=sentence_encoder.encode(X_values_train_flat)
     X_headers_train_embeddings=sentence_encoder.encode(X_headers_train_flat)
     X_values_eval_embeddings=sentence_encoder.encode(X_values_eval_flat)
     X_headers_eval_embeddings=sentence_encoder.encode(X_headers_eval_expanded)

     #label encoding
     label_encoder=LabelEncoder()
     label_encoder.fit(y_train)
     y_train_encoded = label_encoder.transform(y_train_expanded)

     #converting to tensors
     X_values_train_tensor=torch.tensor(X_values_train_embeddings, dtype=torch.float32)
     X_headers_train_tensor=torch.tensor(X_headers_train_embeddings, dtype=torch.float32)
     y_train_tensor=torch.tensor(y_train_encoded, dtype=torch.long)
     X_values_eval_tensor=torch.tensor(X_values_eval_embeddings, dtype=torch.float32)
     X_headers_eval_tensor=torch.tensor(X_headers_eval_embeddings, dtype=torch.float32)

     return X_values_train_tensor, X_headers_train_tensor, y_train_tensor, label_encoder, X_values_eval_tensor, X_headers_eval_tensor, X_headers_eval, X_values_eval

if __name__=="__main__":
     parser=argparse.ArgumentParser(description="Attribute Standardization tool")
     parser.add_argument("csv_file", help="Path to the csv file that contains metadata to be standardized")
     parser.add_argument("--path", choices=["PEPhub_registry", "LOCAL"], default="LOCAL", help="Choose if the CSV file path is a PEPhub registry path or a local path")
     parser.add_argument("--schema", choices=["ENCODE", "FAIRTRACKS"], default="ENCODE", help="Choose the schema to standardize into")
     args=parser.parse_args()
     if args.path == "PEPhub_registry":
          csv_file=standardize_attr_names_from_path(args.csv_file)
          flag="df"
     elif args.path == "LOCAL":
          csv_file=args.csv_file
          flag="path"
     if args.schema == "ENCODE":
          values_file_path = "data/encode_metadata_05022024.csv"
          headers_file_path = "data/encode_headers_matched.csv" 
     elif args.schema == "FAIRTRACKS":
          values_file_path = "data/blueprint_metadata_05012024.csv" 
          headers_file_path = "data/blueprint_metadata_headers_05012024.csv" 
     

     suggestions=standardize_attr_names(csv_file, args.schema, values_file_path, headers_file_path, flag)
     print(suggestions)  


