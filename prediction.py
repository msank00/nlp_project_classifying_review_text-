import warnings
warnings.filterwarnings("ignore")

from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List

from nethub.netsank import Vocabulary
from nethub.netsank import ReviewVectorizer
from nethub.netsank import ReviewDataset
from nethub.netsank import ReviewClassifier
import yaml


from nethub.smutil import *


def predict_rating(args, 
                    review: str, 
                    classifier: ReviewClassifier,
                    vectorizer: ReviewVectorizer, 
                    decision_threshold:float=0.5):
    """Predict the rating of a review
    
    Args:
        review (str): the text of the review
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which separates the rating classes
    """
    # preprocess test review
    review = preprocess_text(review)
    
    vectorized_review = torch.tensor(vectorizer.vectorizer(review)) # dim: [num_features]
    vectorized_review = vectorized_review.view(1, -1) # dim: [1, num_features]
    
    # get raw prediction
    result = classifier(vectorized_review)

    # map to the correct class
    probability_value = F.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index)

def load_vectorizer_for_prediction(file_vectorizer:str):
    """a static method for loading the vectorizer from file

    Args:
        vectorizer_filepath (str): the location of the serialized vectorizer
    Returns:
        an instance of ReviewVectorizer
    """
    with open(file_vectorizer, "r") as fin:
        return ReviewVectorizer.from_serializable(json.load(fin))



# Let's load the yaml file here
with open("config.yaml", 'r') as f:
  config = yaml.load(f)

args = Namespace(frequency_cutoff = config['frequency_cutoff'], 
                 model_state_file=config['model_state_file'],
                 file_csv = config['file_csv'],
                 save_dir= config['save_dir'],
                 vectorizer_file = config['vectorizer_file'],
                 batch_size= config['batch_size'],
                 early_stopping_criteria = config['early_stopping_criteria'],
                 learning_rate = config['learning_rate'],
                 num_epochs = config['num_epochs'],
                 seed = config['seed'],
                 catch_keyboard_interrupt = config['catch_keyboard_interrupt'],
                 cuda = config['cuda'],
                 expand_filepaths_to_save_dir = config['expand_filepaths_to_save_dir'],
                 reload_from_files = config['reload_from_files']
                )

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
    
# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

print("Using CUDA: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)



test_review = "Wonder woman is a beautiful girl"
vectorizer_pred = load_vectorizer_for_prediction(args.vectorizer_file)
classifier_pred = ReviewClassifier(num_features=len(vectorizer_pred.review_vocab))
classifier_pred.load_state_dict(torch.load(args.model_state_file))
classifier_pred = classifier_pred.to(args.device)

prediction = predict_rating(args, test_review, classifier_pred,vectorizer_pred, decision_threshold=0.5)
print("{} -> {}".format(test_review, prediction))


print(">"*10 + "| SUCCESS |" + "<"*10)