import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import time 

input_file="../data/blueprints_unwrapped_metadata.tsv"
start_time_preprocess=time.time()
headers_input="../data/temp_headers.tsv"
headers_df=pd.read_csv(headers_input,sep="\t")
df=pd.read_csv(input_file, sep="\t")
df.replace('NA', np.nan, inplace=True)
for column in df.columns:
    most_common_val = df[column].mode().iloc[0]
    df[column] = df[column].fillna(most_common_val)

#train test validation split 
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_test, df_val = train_test_split(df_temp, test_size=0.5, random_state=42)

#features and labels
X_train = [df_train[column].astype(str).tolist() for column in df_train.columns]
X_train_headers=df_train.columns.tolist()
y_train = df_train.columns
X_test = [df_test[column].astype(str).tolist() for column in df_test.columns]
X_test_headers=df_test.columns.tolist()
y_test = df_test.columns
X_val = [df_val[column].astype(str).tolist() for column in df_val.columns]
X_val_headers=df_val.columns.tolist()
y_val = df_val.columns

# Initialize the sentence transformer model
model_name = 'all-MiniLM-L6-v2' 
sentence_encoder = SentenceTransformer(model_name)

#expanding the headers to match with values
X_train_header_expanded = [header for header in X_train_headers for _ in range(len(df_train))]
X_test_header_expanded = [header for header in X_test_headers for _ in range(len(df_test))]
X_val_header_expanded = [header for header in X_val_headers for _ in range(len(df_val))]

#expanding the labels to match with the inputs (X)
y_train_expanded = [label for label in y_train for _ in range(len(df_train))]
y_test_expanded = [label for label in y_test for _ in range(len(df_test))]
y_val_expanded = [label for label in y_val for _ in range(len(df_val))]

#Flattening column values seuqneces
X_train_flat = [item for sublist in X_train for item in sublist]
X_test_flat = [item for sublist in X_test for item in sublist]
X_val_flat = [item for sublist in X_val for item in sublist]

#sentence transformer embeddings
X_train_embeddings=sentence_encoder.encode(X_train_flat)
X_test_embeddings=sentence_encoder.encode(X_test_flat)
X_val_embeddings=sentence_encoder.encode(X_val_flat)

X_train_header_embeddings=sentence_encoder.encode(X_train_header_expanded)
X_test_header_embeddings=sentence_encoder.encode(X_test_header_expanded)
X_val_header_embeddings=sentence_encoder.encode(X_val_header_expanded)

with open('nn_sentence_transoformer_v5.pkl', 'wb') as f:
    pickle.dump(sentence_encoder, f)

#label encoding
combined_labels = np.concatenate([y_train_expanded, y_test_expanded, y_val_expanded])  
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)
y_train_encoded = label_encoder.transform(y_train_expanded)
y_test_encoded = label_encoder.transform(y_test_expanded)
y_val_encoded = label_encoder.transform(y_val_expanded)

#convert to tensor 
X_train_tensor=torch.tensor(X_train_embeddings, dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_embeddings, dtype=torch.float32)
X_val_tensor=torch.tensor(X_val_embeddings, dtype=torch.float32)

X_train_headers_tensor = torch.tensor(X_train_header_embeddings, dtype=torch.float32)
X_test_headers_tensor = torch.tensor(X_test_header_embeddings, dtype=torch.float32)
X_val_headers_tensor = torch.tensor(X_val_header_embeddings, dtype=torch.float32)

X_train_combined_tensor = torch.cat((X_train_tensor, X_train_headers_tensor), dim=1)
X_test_combined_tensor = torch.cat((X_test_tensor, X_test_headers_tensor), dim=1)
X_val_combined_tensor = torch.cat((X_val_tensor, X_val_headers_tensor), dim=1)

y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

end_time_preprocess=time.time()
time_taken=end_time_preprocess-start_time_preprocess

print("Preprocessing Done.")
print(f"Total time taken for preprocessing:{time_taken:.2f} seconds")