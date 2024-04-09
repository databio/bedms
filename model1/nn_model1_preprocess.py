import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

def load_data(input_file_path):
    """
    TODO change to CSV later 
    Load data from a given tsv file using pandas.
    Impute missing values with the most common value for each column.

    Parameters:
    - input_file_path (str): Path to the input tsv file.

    Returns:
    - df (pd.DataFrame): Processed DataFrame.
    """
    df=pd.read_csv(input_file_path, sep="\t")
    for column in df.columns:
        most_common_val=df[column].mode().iloc[0]
        df[column]=df[column].fillna(most_common_val)
    return df

def split(df):
    """
    Split DataFrame into train, test, and validation sets.
    Reshape the DataFrame.
    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    X_train_2d: Training Feature 2D array 
    X_test_2d : Testing Feature 2D array 
    X_val_2d : Validation Feature 2D array 
    y_train_expanded : Training label array
    y_test_expanded : Testing label array
    y_val_expanded : Validation label array

    """
    df_train,df_temp=train_test_split(df, test_size=0.2, random_state=42)
    df_test,df_val=train_test_split(df_temp, test_size=0.5,random_state=42)
    #features and labels
    X_train=[df_train[column].astype(str).tolist() for column in df_train.columns]
    y_train=df_train.columns
    X_test=[df_test[column].astype(str).tolist() for column in df_test.columns]
    y_test=df_test.columns
    X_val=[df_val[column].astype(str).tolist() for column in df_val.columns]
    y_val=df_val.columns
    #expanding the labels
    y_train_expanded = [label for label in y_train for _ in range(len(df_train))]
    y_test_expanded = [label for label in y_test for _ in range(len(df_test))]
    y_val_expanded = [label for label in y_val for _ in range(len(df_val))]
    #flattening column value sequences
    X_train_flat = [item for sublist in X_train for item in sublist]
    X_test_flat = [item for sublist in X_test for item in sublist]
    X_val_flat = [item for sublist in X_val for item in sublist]
    #reshaping to 2D arrays
    X_train_2d = [[val] for val in X_train_flat]
    X_test_2d = [[val] for val in X_test_flat]
    X_val_2d = [[val] for val in X_val_flat]
    return X_train_2d, y_train_expanded, X_test_2d, y_test_expanded, X_val_2d, y_val_expanded


def to_tensor(encoder_path,X_train_2d, y_train_expanded, X_test_2d, y_test_expanded, X_val_2d, y_val_expanded):
    """
    Preprocess data for model training.
    One Hot Encoding for Features.
    Label Encoding.
    Converting to tensors.
    Parameters:
    - df_train (pd.DataFrame): Training DataFrame.
    - df_test (pd.DataFrame): Test DataFrame.
    - df_val (pd.DataFrame): Validation DataFrame.

    Returns:
    - X_train_tensor (torch.Tensor): Training feature tensor.
    - X_test_tensor (torch.Tensor): Test feature tensor.
    - X_val_tensor (torch.Tensor): Validation feature tensor.
    - y_train_tensor (torch.Tensor): Training label tensor.
    - y_test_tensor (torch.Tensor): Test label tensor.
    - y_val_tensor (torch.Tensor): Validation label tensor.
    """
    #one hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = enc.fit_transform(X_train_2d)
    X_test_encoded = enc.transform(X_test_2d)
    X_val_encoded = enc.transform(X_val_2d)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(enc, f)
    #label encoding 
    combined_labels = np.concatenate([y_train_expanded, y_test_expanded, y_val_expanded])  
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_labels)
    y_train_encoded = label_encoder.transform(y_train_expanded)
    y_test_encoded = label_encoder.transform(y_test_expanded)
    y_val_encoded = label_encoder.transform(y_val_expanded)
    #converting to tensor
    X_train_tensor=torch.tensor(X_train_encoded.toarray(), dtype=torch.float32)
    X_test_tensor=torch.tensor(X_test_encoded.toarray(), dtype=torch.float32)
    X_val_tensor=torch.tensor(X_val_encoded.toarray(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

    torch.save(label_encoder.classes_,encoder_path)
    return combined_labels, X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor


def preprocessing(input_file_path, encoder_path):
    """
    Main Function for data preprocessing.

    Parameters:
    - input_file_path (str): Path to the input TSV file.
    - encoder_file (str): File path to save the OneHotEncoder object.

    Returns:
    - X_train_tensor : Training feature tensor.
    - X_test_tensor : Test feature tensor.
    - X_val_tensor : Validation feature tensor.
    - y_train_tensor : Training label tensor.
    - y_test_tensor : Test label tensor.
    - y_val_tensor : Validation label tensor.
    """
    start_time_preprocess=time.time()
    df=load_data(input_file_path)
    X_train_2d, y_train_expanded, X_test_2d, y_test_expanded, X_val_2d, y_val_expanded=split(df)
    combined_labels, X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor=to_tensor \
        (encoder_path, X_train_2d, y_train_expanded, X_test_2d, y_test_expanded, X_val_2d, y_val_expanded)
    end_time_preprocess=time.time()
    time_taken=end_time_preprocess - start_time_preprocess
    print(f"Total time taken for preprocessing:{time_taken:.2f} seconds")
    return combined_labels, X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor