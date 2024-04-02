# bedmess

bedmess is a tool that can be used to standardize metadata according to the bedbase metadata output schema.

## Models
It has the following models:

### Model 1 - Column Value Based Model
This model trains on the column values only and suggests a column header based on a consensus taken from the prediction of attritbutes for each column. For example, if there are 10 column values for a column, the model will predict 10 attributes first and then provide top three suggested attributes based on the consensus of these 10 predictions.
   The model has 5 associated scripts:
   1. [nn_model1_main.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_main.py) : This is the main script which executes the scripts responsible for data preprocessing, model training and model testing.
      Usage: Assuming you are in the bedmess directory:
      ```bash
      python3 nn_model1_main.py
   2. [nn_model1_preprocess.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_preprocess.py) : This performs data preprocessing and converting to tensors. Presently, we have provided only a dummy file in the data directory. ENCODE and Fairtracks will be provided soon.
   3. [nn_model1_train.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_train.py) : The Neural Network model
   4. [nn_model1_optim.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_optim.py): Hyperparameter tuning
   5. [nn_model1_test.py](https://github.com/databio/bedmess/blob/master/model1/nn_model1_test.py) : This tests the model and generates a confusion matrix, learning curves.

### Model 2 - Column Header and Column Value Based Model
This model trains on both the column values and column headers and suggests a column header based on a consensus taken from the prediction of attributes for each column. There are 5 associated scripts:
   1. [nn_model2_main.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_main.py) : This is the main script. Usage: Considering you are in the model2 directory:
      ```bash
      python3 nn_model2_main.py
   2. [nn_model2_preprocess.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_preprocess.py)
   3. [nn_model2_train.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_train.py)
   4. [nn_model2_optim.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_optim.py)
   5. [nn_model2_test.py](https://github.com/databio/bedmess/blob/master/model2/nn_model2_test.py)

### Model 3 - Bag of Words Encoding Model
This model trains on both column values and column headers. It performs bag of words encoding on both. The output for each column header is top 3 predictions made by the model. There are 4 associated scripts:
   1. [nn_model3_main.py](https://github.com/databio/bedmess/blob/master/model3/nn_model3_main.py) : This is the main script. Usage: Considering you are in the model3 directory:
      ```bash
      python3 nn_model3_main.py
   2. [nn_model3_preprocess.py](https://github.com/databio/bedmess/blob/master/model3/nn_model3_preprocess.py)
   3. [nn_model3_train.py](https://github.com/databio/bedmess/blob/master/model3/nn_model3_train.py)
   4. [nn_model3_test.py](https://github.com/databio/bedmess/blob/master/model3/nn_model3_test.py)

### Model 4 - Bag of Words Encoding + Sentence Transformer Model
This model trains on both column headers and column values. It uses a sentence transformer for creating embeddings for headers and uses bag of words encoding for column values. For each column, it provides the top 3 predictions. There are 4 associated scripts:
   1. [nn_model4_main.py](https://github.com/databio/bedmess/blob/master/model4/nn_model4_main.py) : This is the main script. Usage: Considering you are in the model3 directory:
      ```bash
      python3 nn_model4_main.py
   2. [nn_model4_preprocess.py](https://github.com/databio/bedmess/blob/master/model4/nn_model4_preprocess.py)
   3. [nn_model4_train.py](https://github.com/databio/bedmess/blob/master/model4/nn_model4_train.py)
   4. [nn_model4_test.py](https://github.com/databio/bedmess/blob/master/model4/nn_model4_test.py)

### Model 5 - Sentence Transformer for values and Headers
This model trains on both column headers and values and uses a sentence transformer for creating embeddings for values and headers. For each value in the columns, the model predicts a header and then a consensus is taken for the top 3 predictions for each column. There are 4 associated scripts:
   1. [nn_model5_main.py](https://github.com/databio/bedmess/blob/master/model5/nn_model5_main.py) : This is the main script. Usage: Considering you are in the model5 directory:
      ```bash
      python3 nn_model5_main.py
   2. [nn_model5_preprocess.py](https://github.com/databio/bedmess/blob/master/model5/nn_model5_preprocess.py)
   3. [nn_model5_train.py](https://github.com/databio/bedmess/blob/master/model5/nn_model5_train.py)
   4. [nn_model5_test.py](https://github.com/databio/bedmess/blob/master/model5/nn_model5_test.py)

## Data 
The training data is from tw sources : ENCODE and Blueprints (Fairtracks). Presently, the repo has only 1 dummy dataset. The training data sets will be uploaded soon.

## Usage 
Once you have trained the models by running the commands mentioned above, you can use it for suggesting attributes for your data. The Usage will be updated soon. 
