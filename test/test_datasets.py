# -*- coding: utf-8 -*-
# Created on Mon Mar  9 16:18:14 2020
# @author: arthurd


# PyTorch
import torch
from torchtext import data

from sentipy.datasets import Sentiment140, SSTB


def test_SSTB():
    # Pre processing the Text / Labels
    TEXT = data.Field(batch_first = True)   # you can add "tokenize = 'spacy'" 
    LABEL = data.LabelField(dtype = torch.float)
    
    # Get the training data / validation data and test data
    train_data, test_data = SSTB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(0.8)
    

def test_S140():
    # Pre processing the Text / Labels
    TEXT = data.Field(batch_first = True)   # you can add "tokenize = 'spacy'" 
    LABEL = data.LabelField(dtype = torch.float)
    
    # Get the training data / validation data and test data
    train_data, test_data = Sentiment140.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(0.8)
    print(len(train_data))
        
    
if __name__ == "__main__":
    test_S140()