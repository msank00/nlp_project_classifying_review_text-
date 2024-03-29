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


class Vocabulary(object):
    """Process text and extract vocab for mapping"""

    def __init__(
        self, token_to_idx: Dict = None, add_unk: bool = True, unk_token="<UNK>"
    ):
        """
        Args:
            token_to_idx: a pre-existing map of token to indices
            add_unk: a flag to indicate whether to add unknown tokens
            unk_token: the UNK token to add into the vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def add_token(self, token: str):
        """Update mapping dictionary
        Return:
            the intiger index corresponding to the token
        """
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            return index

    def to_serializable(self):
        """returns a dict that can be serialized"""
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        """instantiates the vocub from a serializable dict"""
        return cls(**contents)

    def add_many(self, tokens: List[str]) -> List[int]:
        """Add list of tokens into the Vocabulary
            
        """
        return [self.add_token(tok) for tok in tokens]

    def lookup_token(self, token: str) -> int:
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError(f"index: {index} not in the Vocabulary")

        return self._idx_to_token[index]

    def __repr__(self):
        return f"Vocabulary Size:{len(self)}"

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    def __init__(self, review_vocab: Vocabulary, rating_vocab: Vocabulary):
        """
        Args:
            review_vocab: maps words to integers
            rating_vocab: maps class labels to integers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorizer(self, review: str):
        """Creates a collapsed one-hot vector for the review
        Args:
            review (str): the review
        Returns:
            one_hot: the collapsed one hot encoding
        """
        # review_vocab: is the entire review vocabulary
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            review_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # add ratings
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # Add top words if count > provided count
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents: Dict):
        """Instantiate a ReviewVectorizer from a serializable dictionary
        
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents["review_vocab"])
        rating_vocab = Vocabulary.from_serializable(contents["rating_vocab"])

        return cls(review_vocab, rating_vocab)

    def to_serializable(self):
        """Create the serializable dictionary for caching
        
        Returns:
            contents (dict): the serializable dictionary
        """
        return {
            "review_vocab": self.review_vocab.to_serializable(),
            "rating_vocab": self.rating_vocab.to_serializable(),
        }


class ReviewDataset(Dataset):
    def __init__(self, review_df: pd.DataFrame, vectorizer: ReviewVectorizer):
        """
        Args:
            review_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_split = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    def set_split(self, split: str = "train"):
        """ selects the splits in the dataset using a column in the dataframe 
        
        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_split[split]

    @classmethod
    def load_dataset_and_make_vectorizer(cls, file_review: str):
        """Load dataset and make a new vectorizer from scratch
        
        Args:
            file_review (str): location of the dataset
        Returns:
            an instance of ReviewDataset
        """
        review_df = pd.read_csv(file_review)
        train_review_df = review_df[review_df.split == "train"]

        return cls(review_df, ReviewVectorizer.from_dataframe(train_review_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, file_review: str, file_vectorizer: str):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use
        
        Args:
            review_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of ReviewDataset
        """
        review_df = pd.read_csv(file_review)
        vectorizer = cls.load_vectorizer_only(file_vectorizer)
        return cls(review_df, vectorizer)

    def save_vectorizer(self, file_vectorizer: str):
        """saves the vectorizer to disk using json
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(file_vectorizer, "w") as fout:
            json.dump(self._vectorizer.to_serializable(), fout)

    @staticmethod
    def load_vectorizer_only(file_vectorizer: str):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(file_vectorizer, "r") as fin:
            return ReviewVectorizer.from_serializable(json.load(fin))

    def get_vectorizer(self):
        """Returns the vectorizer"""
        return self._vectorizer

    def __len__(self):
        return self._target_size

    def __getitem__(self, index: int):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """

        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorizer(row.review)

        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {"x_data": review_vector, "y_target": rating_index}

    def get_num_batches(self, batch_size: int):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


class ReviewClassifier(nn.Module):
    """A simple perseptron based classifier
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features (int): size of the input features
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in: torch.Tensor, apply_sigmoid: bool = False):

        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor
                                x_in.shape = (batch_size, num_features)
            apply_sigmoid (bool): a flag for sigmoid activation
                                should be False if used with cross entropy
                                losses
        Returns:
            the resulting tensor.
            shape: (batch_size,)
        
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)

        return y_out
