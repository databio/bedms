# TODO call optim separately and not within the main script. Only keep training and testing for the main script. 

import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
from nn_model1_train import *
from nn_model1_train import ModelTraining
import pickle
from nn_model1_preprocess import *

param_grid={
    'batch_size':[32, 64, 128],
    'hidden_size':[32,64,128],
    'learning_rate':[0.001,0.005,0.01]
}

param_combinations=ParameterGrid(param_grid)

best_accuracy=0.0
best_params=None
best_train_accuracies=None
best_val_accuracies=None
best_train_losses=None
best_val_losses=None 

for params in param_combinations:
    print("Running on hyperparameters:", params)
    trainer=ModelTraining(input_size, params['hidden_size'], output_size, learning_rate=params['learning_rate'])
    train_accuracies, val_accuracies, train_losses, val_losses=trainer.train(X_train_tensor, y_train_tensor,X_val_tensor, y_val_tensor, num_epochs=10, batch_size=params['batch_size'], device="cpu")
    current_val_accuracy=val_accuracies[-1]
    print("Validation Accuracy:", current_val_accuracy)
    if current_val_accuracy > best_accuracy:
        best_accuracy=current_val_accuracy
        best_params=params
        best_model=trainer.model
        best_train_accuracies=train_accuracies
        best_val_accuracies=val_accuracies
        best_train_losses=train_losses
        best_val_losses=val_losses

print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)

best_model_path="nn_model1_best.pth"
torch.save(best_model.state_dict(), best_model_path)

results={'best_hyperparameters': best_params, 'best_train_accuracies': best_train_accuracies, 'best_val_accuracies':best_val_accuracies,
         'best_train_losses':best_train_losses, 'best_val_losses':best_val_losses}

with open('hyperparam_optim_results.pkl', 'wb') as f:
    pickle.dump(results, f)
    


