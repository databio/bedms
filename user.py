import argparse
import pickle

#arguements from user - inout data & model selection 
# TODO output schema selection - blueprints/ encode 
def main():
    parser=argparse.ArgumentParser(description='Metadata Standarization')
    parser.add_argument('--model-name', type=str, help='Model to be used for suggestions', required=True, choices=['model1', 'model2', 'model3', 'model4', 'model5'])
    parser.add_argument('--input-file', type=str,help='Path to input file - presently taking tsv', required=True)
    args=parser.parse_args()
    
#preprocessing user input based on each model 

#model input parameters - fetch the best hyperparameters from the optim output files, separate for each model
#Each model will have two options - encode and fairtracks
#predictions - separate for each model

#decode predicted labels - common

#print predictions - separate for each model 
#outputs will be in a different format and suggestions list to have over 50% votes suggestions only- 
'''
{
    'old_attribute': {'suggestion_1: <probability>, 'suggestion_2': <probability>},
    'old_attribute_2':{'suggestion_1: <probability>, 'suggestion_2': <probability>}
}

'''

import peppy

project = peppy.Project("/home/saanika/example_peps/example_basic/project_config.yaml") # instantiate in-memory Project representation
samples = project.samples # grab the list of Sample objects defined in this Project

# Find the input file for the first sample in the project
print(samples[0]["file"])
