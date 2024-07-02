import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import json
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging 
import os
from glob import glob
from torch.utils.data import Dataset
import random
from torch.nn.utils.rnn import pad_sequence
from memory_profiler import profile
import optuna
import pickle 


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
random.seed(seed)
np.random.seed(seed)

#logging set up
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

##NN model - Bag of Words + Sentence Transformer Model
class BoWSTModel(nn.Module):
    def __init__(self, input_size_values, input_size_values_embeddings, input_size_headers, hidden_size, output_size, dropout_prob):
        super(BoWSTModel, self).__init__()
        self.fc_values1 = nn.Linear(input_size_values, hidden_size)
        self.dropout_values1 = nn.Dropout(dropout_prob)
        self.fc_values2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_values2 = nn.Dropout(dropout_prob)
        self.fc_values_embeddings1=nn.Linear(input_size_values_embeddings, hidden_size)
        self.dropout_values_embeddings1=nn.Dropout(dropout_prob)
        self.fc_values_embeddings2=nn.Linear(hidden_size,hidden_size)
        self.dropout_values_embeddings2=nn.Dropout(dropout_prob)
        self.fc_headers1 = nn.Linear(input_size_headers, hidden_size)
        self.dropout_headers1 = nn.Dropout(dropout_prob)
        self.fc_headers2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_headers2 = nn.Dropout(dropout_prob)
        self.fc_combined1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout_combined1 = nn.Dropout(dropout_prob)
        self.fc_combined2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_values, x_values_embeddings, x_headers):
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


def load_from_dir(dir):
    """
    Loads each file from the directory path.
    Args:
    dir (str): Path to the directory.
    Returns:
    str: paths to each file in the directory. 
    """
    return glob(os.path.join(dir, "*.csv"))

def load_and_preprocess(file_path):
    """
    Loads and Preprocesses each csv file as a Pandas DataFrame.
    Args:
    file_path (str): Path to each csv file.
    Returns:
    pandas.DataFrame: df of each csv file. 
    """
    df = pd.read_csv(file_path, sep=",")
    df.replace('NA', np.nan, inplace=True)
    for column in df.columns:
        most_common_val = df[column].mode().iloc[0]
        df[column] = df[column].fillna(most_common_val)
    return df

@profile
def accumulate_data(files):
    """
    Accumulates data from multiple files into lists.
    Args:
    files (nested list): List containing sublists of values or header files.
    Returns:
    tuple of lists: Lists of values, headers, labels.
    """
    X_values_list=[]
    X_headers_list=[]
    y_list=[]
    for values_file, headers_file in files:
        df_values=load_and_preprocess(values_file)
        df_headers=load_and_preprocess(headers_file)
        y = df_values.columns
        table_list=[]
        #values list
        for col in df_values.columns:
            sublist_list=df_values[col].tolist()
            table_list.append(sublist_list)
        X_values_list.append(table_list)
        #headers list
        table_list=[]
        for col in df_headers.columns:
            sublist_list=df_headers[col].tolist()
            table_list.append(sublist_list)
        X_headers_list.append(table_list)
        #y list
        y_list.append(y)
    return X_values_list, X_headers_list, y_list

def get_top_k_average(embeddings, k=3):
    """
    Computes the average of the top k=3 most frequent embeddings. 
    Args:
    embeddings (list of tensors): List of embedding tensors of a column.
    k (int) : Number of the top frequent embeddings to evaluate on.
    Returns:
    torch.Tensor: Averaged embedding of the top k embeddings in each column of the metadata. 
    """
    flattened = [tuple(embedding.tolist()) for embedding in embeddings]
    counter = Counter(flattened)
    top_k = counter.most_common(k)

    top_k_embeddings = [torch.tensor(item[0]) for item in top_k]
    top_k_average = torch.mean(torch.stack(top_k_embeddings), dim = 0)

    return top_k_average

def lazy_loading(data_list, batch_size):
    """
    Lazy loading for data in batches. 
    Args:
    data_list(list); List of data to be loaded lazily.
    batch_size(int) : Size of batch.
    Yield:
    list : Batch of the data. 
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i+batch_size]

@profile
def encoding(X_values_train_list, X_headers_train_list, y_train_list, \
             X_values_test_list, X_headers_test_list, y_test_list, \
                  X_values_val_list, X_headers_val_list, y_val_list ):
    """
    Encodes data using Bag of Words, Sentence Transformers, and label encoding.
    Args:
    X_values_train_list, X_headers_train_list, y_train_list: list of training data.
    X_values_test_list, X_headers_test_list, y_test_list: list of test data.
    X_values_val_list, X_headers_val_list, y_val_list: list of validation data.
    Returns:
    list of torch.Tensor : tensor lists of the encoded training, test, and validation data. 
    """
    #Bag of Words
    #Flatten the nested lists
    #Reducing the vocabulary size - limiting the number of tables that are considered
    X_values_train_list_top = X_values_train_list[:500]
    flattened_list = [item for sublist in X_values_train_list_top for col in sublist for item in col]
    vectorizer=CountVectorizer()
    vectorizer.fit(flattened_list)
    with open('vectorizer_new.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    #Sentence Transformers
    model_name = 'all-MiniLM-L6-v2'
    sentence_encoder = SentenceTransformer(model_name)

    #Label Encoders
    label_encoder=LabelEncoder()
    flat_y_train = [','.join(y) for y in y_train_list]
    individual_values = [value.strip() for y in flat_y_train for value in y.split(',')]
    unique_values = set(individual_values)
    unique_values_list = list(unique_values)
    label_encoder.fit(unique_values_list)

    label_encoder_path="label_encoder_model6_new.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder,f)

    def encode_data(X_values_list, X_headers_list, y_list, batch_size):
        X_values_bow_tensor_list, X_values_st_tensor_list, X_headers_st_tensor_list, y_tensor_list =[], [], [], []
        data_batches=lazy_loading(list(zip(X_values_list, X_headers_list, y_list)), batch_size)
        for data_batch in data_batches:
            for X_values, X_headers, y in data_batch:
                X_values_bow = [vectorizer.transform(col).toarray().flatten() for col in X_values]
                X_values_bow_tensor_list.append(torch.tensor(X_values_bow, dtype=torch.float32))

                X_values_st = [sentence_encoder.encode(col, show_progress_bar = False) for col in X_values]
                averaged_X_values_st = [get_top_k_average([torch.tensor(embedding, dtype=torch.float32) for embedding in col]) for col in X_values_st]
                X_values_st_tensor_list.append(torch.stack(averaged_X_values_st))

                X_headers_st = [sentence_encoder.encode(col, show_progress_bar = False) for col in X_headers]
                averaged_X_headers_st = [get_top_k_average([torch.tensor(embedding, dtype=torch.float32) for embedding in col]) for col in X_headers_st]
                X_headers_st_tensor_list.append(torch.stack(averaged_X_headers_st))

                categories = list(y)
                y_encoded = [label_encoder.transform([cat]) for cat in categories]
                y_tensor = torch.tensor(y_encoded, dtype=torch.long)
                y_tensor_list.append(y_tensor)

            y_tensor=y_tensor_list

            return X_values_bow_tensor_list, X_values_st_tensor_list, X_headers_st_tensor_list, y_tensor
    batch_size = 32
    X_values_train_bow_tensor, X_values_train_st_tensor, X_headers_train_tensor, y_train_tensor=encode_data(X_values_train_list, X_headers_train_list, y_train_list, batch_size)
    X_values_test_bow_tensor, X_values_test_st_tensor, X_headers_test_tensor, y_test_tensor = encode_data(X_values_test_list, X_headers_test_list, y_test_list, batch_size)
    X_values_val_bow_tensor, X_values_val_st_tensor, X_headers_val_tensor, y_val_tensor = encode_data(X_values_val_list, X_headers_val_list, y_val_list, batch_size)

    return X_values_train_bow_tensor, X_values_train_st_tensor, X_headers_train_tensor, y_train_tensor, X_values_test_bow_tensor, \
          X_values_test_st_tensor, X_headers_test_tensor, y_test_tensor, \
              X_values_val_bow_tensor, X_values_val_st_tensor, X_headers_val_tensor, y_val_tensor, label_encoder, unique_values_list

@profile
def data_loader(X_values_bow_tensor_list, X_values_st_tensor_list, X_headers_st_tensor_list, y_tensor_list, batch_size):
    """
    Creates PyTorch DataLoader for the input data.
    Args:
    X_values_bow_tensor_list (list of torch.Tensor): List of tensors containing Bag of Words representations of the values.
    X_values_st_tensor_list (list of torch.Tensor): List of tensors containing Sentence Transformer Embeddings of the values.
    X_headers_st_tensor_list (list of torch.Tensor): List of tensors containing Sentence Transformer EMbeddings of the headers.
    y_tensor_list (list of torch.Tensor): List of tensors containing the label encoded labels.
    batch_size (int): Batch Size of the DataLoader
    Returns:
    DataLoader: PyTorch DataLoader object containing the combined dataset. 

    """
    datasets=[]
    for X_values_bow_tensor, X_values_st_tensor, X_headers_st_tensor, y_tensor in zip(X_values_bow_tensor_list, X_values_st_tensor_list, X_headers_st_tensor_list, y_tensor_list):
        if y_tensor.dim() > 1:
            y_tensor = y_tensor.view(-1)
        dataset = TensorDataset(X_values_bow_tensor, X_values_st_tensor, X_headers_st_tensor, y_tensor)
        datasets.append(dataset)
    combined_dataset=torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=batch_size, pin_memory=True)
    return loader

@profile
def custom_train(model, criterion, optimizer, train_loader, val_loader, num_epochs, checkpoint_path):
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    best_val_accuracy = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        indices = list(range(len(train_loader.dataset)))
        random.shuffle(indices)
        shuffled_dataset = torch.utils.data.Subset(train_loader.dataset, indices)
        train_loader = DataLoader(shuffled_dataset, batch_size = train_loader.batch_size)
        # Training phase
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for X_values_bow, X_values_st, X_headers_st, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X_values_bow, X_values_st, X_headers_st)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train 
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_values_bow, X_values_st, X_headers_st, y in val_loader:
                outputs = model(X_values_bow, X_values_st, X_headers_st)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if val_accuracy > best_val_accuracy :
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            # TODO : save the model
            torch.save({'epoch': best_epoch, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,}, checkpoint_path)

        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    torch.save(model.state_dict(), "model6_red_new.pth")
    return train_accuracies, val_accuracies, train_losses, val_losses
        

def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_values_bow, X_values_st, X_headers_st, y in test_loader:
            outputs = model(X_values_bow, X_values_st, X_headers_st)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted ==y).sum().item()
            y_true.extend(y.tolist())
            y_pred.extend(predicted.tolist())
    test_loss /= len(test_loader)
    accuracy = 100 * correct/total
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average = "weighted")
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    return y_true, y_pred, outputs

def plot_confusion_matrix(y_true, y_pred, unique_values_list):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(unique_values_list), yticklabels=np.unique(unique_values_list))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix_model6_mt_new.jpg")
    plt.show()
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {i}: {acc:.4f}")

def plot_learning_curve(num_epochs, train_accuracies, val_accuracies, train_losses, val_losses):
    #accuracy
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_model6_mt_new.jpg")
    plt.show()
    #loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses , label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_model6_mt_new.jpg")
    plt.show()

def hyperparameter_tuning(trial):
    hidden_size= trial.suggest_categorical('hidden_size', [32,64,128,256,512])
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    input_size_values = X_values_train_bow_tensor_list[0].size(1)
    input_size_values_embeddings = X_values_train_st_tensor_list[0].size(1)
    input_size_headers = X_headers_train_tensor_list[0].size(1)
    output_size = len(label_encoder.classes_) 

    model = BoWSTModel(input_size_values, input_size_values_embeddings, input_size_headers, hidden_size, output_size, dropout_prob)
    l2_reg_lambda = 0.001
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=l2_reg_lambda)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    train_loader = data_loader(X_values_train_bow_tensor_list, X_values_train_st_tensor_list, X_headers_train_tensor_list, y_train_tensor_list, batch_size)
    val_loader = data_loader(X_values_val_bow_tensor_list, X_values_val_st_tensor_list, X_headers_val_tensor_list, y_val_tensor_list, batch_size)

    num_epochs = 20
    train_accuracies, val_accuracies, train_losses, val_losses = custom_train(model, criterion, optimizer, train_loader, val_loader, num_epochs)
    mean_val_accuracy = np.mean(val_accuracies)
    trial.set_user_attr('train_accuracies', train_accuracies)
    trial.set_user_attr('val_accuracies', val_accuracies)
    trial.set_user_attr('train_losses', train_losses)
    trial.set_user_attr('val_losses', val_losses)

    trial.set_user_attr('mean_val_accuracy', mean_val_accuracy)
    trial.set_user_attr('best_epoch', val_accuracies.index(max(val_accuracies)) + 1)

    print(f"Trial {trial.number}, hidden_size: {hidden_size}, dropout_prob: {dropout_prob}, learning_rate: {learning_rate}, mean_val_accuracy: {mean_val_accuracy:.4f}")

    return mean_val_accuracy
                        
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

if __name__=="__main__":
    values_dir="/scratch/bam8pm/mini_files_new/"
    headers_dir="/scratch/bam8pm/mini_headers_new/"
    #load the files from dir
    values_files=load_from_dir(values_dir)
    headers_files=load_from_dir(headers_dir)
    #pairing the files together
    paired_files = list(zip(values_files, headers_files))
    # TODO It is not pairing th files number wise, check for discrepancy 
    #print(paired_files)
    #paired shuffle
    random.shuffle(paired_files)
    train_files = paired_files[:1000]
    test_files = paired_files[800:900]
    val_files = paired_files[900:1000]

    #generate lists for train, test, val 
    X_values_train_list, X_headers_train_list, y_train_list = accumulate_data(train_files)
    X_values_test_list, X_headers_test_list, y_test_list = accumulate_data(test_files)
    X_values_val_list, X_headers_val_list, y_val_list = accumulate_data(val_files)
    logger.info("Accumulation Done.")

    #encoding/embedding
    X_values_train_bow_tensor_list, X_values_train_st_tensor_list, X_headers_train_tensor_list, y_train_tensor_list, X_values_test_bow_tensor_list, \
          X_values_test_st_tensor_list, X_headers_test_tensor_list, y_test_tensor_list, \
              X_values_val_bow_tensor_list, X_values_val_st_tensor_list, X_headers_val_tensor_list, y_val_tensor_list, label_encoder, unique_values_list= encoding(X_values_train_list, X_headers_train_list, y_train_list, \
             X_values_test_list, X_headers_test_list, y_test_list, \
                  X_values_val_list, X_headers_val_list, y_val_list )
    logger.info("Encoding Done.")

    #data loader
    batch_size=32
    train_loader = data_loader(X_values_train_bow_tensor_list, X_values_train_st_tensor_list, X_headers_train_tensor_list, y_train_tensor_list, batch_size)
    test_loader = data_loader(X_values_test_bow_tensor_list, X_values_test_st_tensor_list, X_headers_test_tensor_list, y_test_tensor_list, batch_size)
    val_loader = data_loader(X_values_val_bow_tensor_list, X_values_val_st_tensor_list, X_headers_val_tensor_list, y_val_tensor_list, batch_size)
    '''
    #hyperparameter tuning
    study = optuna.create_study(direction = 'maximize')
    study.optimize(hyperparameter_tuning, n_trials = 10)
    best_params = study.best_params
    print("Best Hyperparameters are:", best_params)
    best_trial = study.best_trial
    print(f"Trial number: {best_trial.number}")
    print(f"Hidden Size: {best_trial.params['hidden_size']}")
    print(f"Dropout Probability: {best_trial.params['dropout_prob']}")
    print(f"Learning Rate: {best_trial.params['learning_rate']}")
    print(f"Mean Validation Accuracy: {best_trial.user_attrs['mean_val_accuracy']:.4f}")
    print(f"Best Epoch: {best_trial.user_attrs['best_epoch']}")

    # Logging all trials
    for trial in study.trials:
        trial_num = trial.number
        hidden_size = trial.params['hidden_size']
        dropout_prob = trial.params['dropout_prob']
        learning_rate = trial.params['learning_rate']
        mean_val_accuracy = trial.user_attrs['mean_val_accuracy']
        best_epoch = trial.user_attrs['best_epoch']
        print(f"Trial {trial_num}, hidden_size: {hidden_size}, dropout_prob: {dropout_prob}, learning_rate: {learning_rate}, mean_val_accuracy: {mean_val_accuracy:.4f}, best_epoch: {best_epoch}")

    '''
    #Model Initialization
    input_size_values = X_values_train_bow_tensor_list[0].size(1)
    input_size_values_embeddings = X_values_train_st_tensor_list[0].size(1)
    input_size_headers = X_headers_train_tensor_list[0].size(1)
    hidden_size = 64
    output_size = len(label_encoder.classes_)
    dropout_prob=0.13
    learning_rate = 0.0075

    model = BoWSTModel(input_size_values, input_size_values_embeddings, input_size_headers, hidden_size, output_size, dropout_prob)
    criterion = nn.CrossEntropyLoss()
    l2_reg_lambda = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg_lambda)
    #Training the model
    checkpoint_path = "checkpoint_new.pt"
    num_epochs = 20
    total_params = count_parameters(model)
    print(f"Total number of parameters in the model: {total_params}")
    train_accuracies, val_accuracies, train_losses, val_losses = custom_train(model, criterion, optimizer, train_loader, val_loader, num_epochs, checkpoint_path)

    #Testing the model
    y_true, y_pred, outputs = test_model(model, criterion, test_loader)
    deencoded_labels = label_encoder.inverse_transform(y_true)
    deencoded_preds = label_encoder.inverse_transform(y_pred)
    print(deencoded_labels)
    print(deencoded_preds)
    #plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, unique_values_list)
    #plot learning curve
    plot_learning_curve(num_epochs, train_accuracies, val_accuracies, train_losses, val_losses)