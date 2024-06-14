import pandas as pd
import numpy as np
import logging
import argparse
from pephubclient import PEPHubClient
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pickle
from collections import Counter

#logging set up
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class sentence_transformer_NN(nn.Module):
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
        super(sentence_transformer_NN, self).__init__()
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

def load_data(csv_file_path, flag):
    """
    Load data from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    if flag=="path":
        df=pd.read_csv(csv_file_path, sep=",", dtype=str)
    elif flag=="df":
        df=csv_file_path
    df.replace('NA', np.nan, inplace=True)
    for column in df.columns:
        if df[column].notna().any(): #skipping those columns that are all empty 
            most_common_val = df[column].mode().iloc[0]
            #print(most_common_val)
            df[column] = df[column].fillna(most_common_val)
    return df 

def data_preprocessing(csv_file, flag, label_encoder_pth):
    """
    Preprocess the data for training and evaluation.
    Args:
        csv_file (str): Path to the csv file containing user provided metadata.
    Returns:
        tuple: Processed data tensors
        pandas.DataFrame: User provided metadata .
    """
    df_user = load_data(csv_file, flag)
    X_values=[df_user[column].astype(str).tolist() for column in df_user.columns]
    X_headers=df_user.columns
    X_values_flat=[item for sublist in X_values for item in sublist]
    X_headers_expanded=[x for x in X_headers for _ in range(len(df_user))]

    #Sentence Transformer Embeddings
    model_name='all-MiniLM-L6-v2' 
    sentence_encoder = SentenceTransformer(model_name)
    X_values_embeddings=sentence_encoder.encode(X_values_flat, show_progress_bar = False)
    X_headers_embeddings=sentence_encoder.encode(X_headers_expanded, show_progress_bar = False)

    #Label Encoding
    label_encoder = LabelEncoder()
    with open(label_encoder_pth, 'rb') as f:
        label_encoder = pickle.load(f)
    
    #Converting to tensors
    X_values_tensor=torch.tensor(X_values_embeddings, dtype=torch.float32)
    X_headers_tensor=torch.tensor(X_headers_embeddings, dtype=torch.float32)

    return X_values_tensor, X_headers_tensor, label_encoder, X_headers

def standardize_attr_names(csv_file, schema, model_pth, label_encoder_pth, flag):
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
    X_values_tensor, X_headers_tensor, label_encoder, X_headers = data_preprocessing(csv_file, flag, label_encoder_pth)
    logger.info("Data Preprocessing completed.")
    input_size_values = X_values_tensor.shape[1]
    input_size_headers = X_headers_tensor.shape[1]
    hidden_size = 256
    output_size = len(label_encoder.classes_)
    model = sentence_transformer_NN(input_size_values, input_size_headers, hidden_size, output_size, dropout_prob=0.5)
    model=torch.load(model_pth)
    model.eval()

    all_preds =[]
    with torch.no_grad():
        outputs = model(X_values_tensor, X_headers_tensor)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
    
    decoded_predictions = label_encoder.inverse_transform(all_preds)
    num_categories = len(X_headers)
    num_predictions=len(decoded_predictions)
    predictions_per_category = num_predictions // num_categories
    grouped_preds = [decoded_predictions[i*predictions_per_category : (i+1)*predictions_per_category] for i in range(num_categories)]

    suggestions={}
    for header, preds in zip(X_headers, grouped_preds):
        pred_counts=Counter(preds)
        total_predictions=len(preds)
        probabilities = {attr: count / total_predictions for attr, count in pred_counts.items() if count/total_predictions>0.5}
        suggestions[header]=probabilities
    return suggestions
        
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
    if args.schema == "FAIRTRACKS":
        model_pth = "model5/trained_model_fairtracks.pth"
        label_encoder_pth = "model5/label_encoder_fairtracks.pkl"
    elif args.schema == "ENCODE":
        model_pth = None
        label_encoder_pth = None
    suggestions = standardize_attr_names(csv_file, args.schema, model_pth, label_encoder_pth, flag)
    print(suggestions)