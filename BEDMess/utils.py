import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter 
from huggingface_hub import hf_hub_download

def data_preprocessing(df):
    X_values_st = [df[column].astype(str).tolist() for column in df.columns]
    X_headers_st = df.columns.tolist()
    X_values_bow = [df[column].astype(str).tolist() for column in df.columns]

    return X_values_st, X_headers_st, X_values_bow

def get_top_k_average(val_embedding, k):
    embeddings_list = [tuple(embedding) for embedding in val_embedding]
    counts = Counter(embeddings_list)
    top_3_embeddings = [np.array(embedding) for embedding, count in counts.most_common(k)]
    top_3_embeddings_tensor = torch.tensor(top_3_embeddings)
    column_embedding_mean = torch.mean(top_3_embeddings_tensor, dim=0)

    return column_embedding_mean.numpy()

def data_encoding(X_values_st, X_headers_st, X_values_bow, schema):
    #Sentence Transformer Model
    model_name='all-MiniLM-L6-v2' 
    sentence_encoder = SentenceTransformer(model_name)
    X_headers_embeddings=sentence_encoder.encode(X_headers_st, show_progress_bar = False) 
    #generating embeddings for each element in sublist (column)
    embeddings = []
    for column in X_values_st:
        val_embedding = sentence_encoder.encode(column, show_progress_bar = False)
        embedding = get_top_k_average(val_embedding, k=3)
        embeddings.append(embedding)
    X_values_embeddings = embeddings
    if schema == "ENCODE":
        #Bag of Words Vectorizer
        vectorizer=CountVectorizer()
        vc_path = hf_hub_download(repo_id="databio/attribute-standardizer-model6", filename="vectorizer_encode.pkl")
        with open(vc_path, "rb") as f:
            vectorizer=pickle.load(f)
        transformed_columns = []
        for column in X_values_bow:
            column_text = ' '.join(column)
            transformed_column = vectorizer.transform([column_text])
            transformed_columns.append(transformed_column.toarray()[0])
        transformed_columns = np.array(transformed_columns)
        #print(transformed_columns)
        X_values_bow = transformed_columns
        #Label Encoding
        # TODO change this to the new one 
        label_encoder = LabelEncoder()
        lb_path = hf_hub_download(repo_id="databio/attribute-standardizer-model6", filename = "label_encoder_encode.pkl")
        with open(lb_path, 'rb') as f:
            label_encoder = pickle.load(f) 

    elif schema == "FAIRTRACKS":
        raise NotImplementedError
    
    X_headers_embeddings_tensor = torch.tensor(X_headers_embeddings, dtype = torch.float32)
    X_values_embeddings_tensor = torch.tensor(X_values_embeddings, dtype = torch.float32)
    X_values_bow_tensor = torch.tensor(X_values_bow, dtype = torch.float32)

    return X_headers_embeddings_tensor, X_values_embeddings_tensor, X_values_bow_tensor, label_encoder






    


