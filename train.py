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


if args.reload_from_files:
    # training from a checkpoint
    print("Loading dataset and vectorizer")
    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.file_csv,
                                                            args.vectorizer_file)
else:
    print("Loading dataset and creating vectorizer")
    # create dataset and vectorizer
    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.file_csv)
    dataset.save_vectorizer(args.vectorizer_file) 


vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))

classifier = classifier.to(args.device)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), 
                       lr=args.learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                 mode = 'min', 
                                                 factor = 0.5, 
                                                 patience = 1)

train_state = make_train_state(args)




try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        
        # Iterate over training dataset
        # setup: batch generator, set loss and acc to 0, set train mode on
        
        dataset.set_split("train")
        batch_generator = generate_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device = args.device)

        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            
            optimizer.zero_grad()
            y_pred = classifier(x_in=batch_dict['x_data'].float())
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            loss.backward()
            optimizer.step()
            
            # compute accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            status = f"train\t epoch:{epoch_index}/{args.num_epochs}\t batch_index:{batch_index}\t train_acc: {running_acc:.4}\t train_loss: {running_loss:.4}"
            print(status)
            
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('val')
        batch_generator = generate_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            
            
            status = f"val\t epoch:{epoch_index}/{args.num_epochs}\t batch_index:{batch_index}\t val_acc: {running_acc:.4}\t val_loss: {running_loss:.4}"
            print(status)
            

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=args, model=classifier,
                                         train_state=train_state)

        scheduler.step(train_state['val_loss'][-1])

        
        if train_state['stop_early']:
            break
        
except KeyboardInterrupt:
    print("Exiting loop")


classifier.load_state_dict(torch.load(train_state['model_filename']))
classifier = classifier.to(args.device)

dataset.set_split('test')
batch_generator = generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in tqdm(enumerate(batch_generator)):
    # compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss: {:.3f}".format(train_state['test_loss']))
print("Test Accuracy: {:.2f}".format(train_state['test_acc']))


print("==SUCCESS==")