import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

input_file_path = "data/dummy_1.tsv" 
df = pd.read_csv(input_file_path, sep="\t")

df.replace('NA', np.nan, inplace=True)

for column in df.columns:
    most_common_val = df[column].mode().iloc[0]
    df[column] = df[column].fillna(most_common_val)

# Train-test-validation split
df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_test, df_val = train_test_split(df_temp, test_size=0.5, random_state=42)

#features and labels
X_train = [df_train[column].astype(str).tolist() for column in df_train.columns]
y_train = df_train.columns
X_test = [df_test[column].astype(str).tolist() for column in df_test.columns]
y_test = df_test.columns
X_val = [df_val[column].astype(str).tolist() for column in df_val.columns]
y_val = df_val.columns

#expanding the labels to match with the inputs (X)
y_train_expanded = [label for label in y_train for _ in range(len(df_train))]
y_test_expanded = [label for label in y_test for _ in range(len(df_test))]
y_val_expanded = [label for label in y_val for _ in range(len(df_val))]

#Flattening column values seuqneces
X_train_flat = [item for sublist in X_train for item in sublist]
X_test_flat = [item for sublist in X_test for item in sublist]
X_val_flat = [item for sublist in X_val for item in sublist]

#Reshaping to 2D arrays
X_train_2d = [[val] for val in X_train_flat]
X_test_2d = [[val] for val in X_test_flat]
X_val_2d = [[val] for val in X_val_flat]

# One-hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = enc.fit_transform(X_train_2d)
X_test_encoded = enc.transform(X_test_2d)
X_val_encoded = enc.transform(X_val_2d)
with open('dummy_encoder.pkl', 'wb') as f:
    pickle.dump(enc, f)

#label encoding
combined_labels = np.concatenate([y_train_expanded, y_test_expanded, y_val_expanded])  
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)
y_train_encoded = label_encoder.transform(y_train_expanded)
y_test_encoded = label_encoder.transform(y_test_expanded)
y_val_encoded = label_encoder.transform(y_val_expanded)

#converting to tensors
X_train_tensor=torch.tensor(X_train_encoded.toarray(), dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_encoded.toarray(), dtype=torch.float32)
X_val_tensor=torch.tensor(X_val_encoded.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

print("Preprocessing Done.")
