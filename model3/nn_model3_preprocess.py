import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

input_file_path="../data/dummy_1.tsv"
df = pd.read_csv(input_file_path, sep="\t")
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

#Joining to form a single string
#NOTE : Each string in the list is a column and contains all col values
X_train_strings = [" ".join([str(val) for val in sublist]) for sublist in X_train] 
X_test_strings = [" ".join([str(val) for val in sublist]) for sublist in X_test]
X_val_strings = [" ".join([str(val) for val in sublist]) for sublist in X_val]

X_train_header_strings = [" ".join([header for _ in range(len(df_train))]) for header in X_train_headers]
X_test_header_strings = [" ".join([header for _ in range(len(df_test))]) for header in X_test_headers]
X_val_header_strings = [" ".join([header for _ in range(len(df_val))]) for header in X_val_headers]

#count vectorizer
vectorizer = CountVectorizer()
all_data=X_train_strings+X_train_header_strings
vectorizer.fit(all_data)
X_train_bow = vectorizer.transform(X_train_strings)
X_train_header_bow = vectorizer.transform(X_train_header_strings)
X_test_bow = vectorizer.transform(X_test_strings)
X_test_header_bow = vectorizer.transform(X_test_header_strings)
X_val_bow = vectorizer.transform(X_val_strings)
X_val_header_bow = vectorizer.transform(X_val_header_strings)
#saving the vectorizer
with open("vectorizer_v3.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

#label encoding for y
label_encoder=LabelEncoder()
combined_labels = np.concatenate([y_train, y_test, y_val])
label_encoder.fit(combined_labels)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_val_encoded = label_encoder.transform(y_val)

#converting to tensors
X_train_bow_tensor = torch.tensor(X_train_bow.toarray(), dtype=torch.float32)
X_test_bow_tensor = torch.tensor(X_test_bow.toarray(), dtype=torch.float32)
X_val_bow_tensor = torch.tensor(X_val_bow.toarray(), dtype=torch.float32)

X_train_header_bow_tensor = torch.tensor(X_train_header_bow.toarray(), dtype=torch.float32)
X_test_header_bow_tensor = torch.tensor(X_test_header_bow.toarray(), dtype=torch.float32)
X_val_header_bow_tensor = torch.tensor(X_val_header_bow.toarray(), dtype=torch.float32)

X_train_combined_tensor = torch.cat((X_train_bow_tensor, X_train_header_bow_tensor), dim=1)
X_test_combined_tensor = torch.cat((X_test_bow_tensor, X_test_header_bow_tensor), dim=1)
X_val_combined_tensor = torch.cat((X_val_bow_tensor, X_val_header_bow_tensor), dim=1)

y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

print("Preprocessing Done.")