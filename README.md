# bedmess

bedmess is a tool that can be used to standardize metadata according to the bedbase metadata output schema.

## Models
It has the following models:

### Model 1 - Column Value Based Model
This model trains on the column values only and suggests a column header based on a consensus taken from the prediction of attritbutes for each column. For example, if there are 10 column values for a column, the model will predict 10 attributes first and then provide top three suggested attributes based on the consensus of these 10 predictions.
   The model has 4 associated scripts:
   1. [nn_model1_main.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_main.py) : This is the main script which executes the scripts responsible for data preprocessing, model training and model testing.
      Usage: Assuming you are in the bedmess directory:
      ```bash
      python3 nn_model1_main.py
   2. [nn_model1_preprocess.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_preprocess.py) : This performs data preprocessing and converting to tensors. Presently, we have provided only a dummy file in the data directory. ENCODE and Fairtracks will be provided soon.
   3. [nn_model1_train.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_train.py) : The Neural Network model
   4. [nn_model1_test.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_test.py) : This tests the model and generates a confusion matrix, learning curves.

### Model 2 - Column Header and Column Value Based Model
This model trains on both the column values and column headers and suggests a column header based on a consensus taken from the prediction of attributes for each column. There are 4 associated scripts:
   1. [nn_model2_main.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_main.py) : This is the main script. Usage: Considering you are in the model2 directory:
      ```bash
      python3 nn_model2_main.py
   2. [nn_model2_preprocess.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_preprocess.py)
   3. [nn_model2_train.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_train.py)
   4. [nn_model2_test.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_test.py)
