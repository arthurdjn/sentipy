# -*- coding: utf-8 -*-
# Created on Mon Mar  9 15:36:08 2020
# @author: arthurd


# PyTorch
import torch
from torchtext import data
import torch.nn.functional as F

# SentiPy modules
from sentipy.model.model import CNN
from sentipy.dataset.ssb import SSB
from sentipy.utils import tokenizer


def process(tokenize = None,
            batch_size = 64,
            vocab_size = 20000):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    
    # Pre processing the Text / Labels
    TEXT = data.Field(tokenize = tokenize, batch_first = True)   # you can add "tokenize = 'spacy'" 
    LABEL = data.LabelField(dtype = torch.float)
    
    # 1/ Get the training data / validation data and test data
    train_data, test_data = SSB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(0.8)
    
    # 2/ Create the vocabulary for words embeddings
    TEXT.build_vocab(train_data, 
                     max_size = vocab_size, 
                     vectors = "glove.6B.100d", 
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    
    # 3/ Create iterators
    train_iterator = data.BucketIterator(train_data, 
                                         shuffle = True,
                                         batch_size = batch_size, 
                                         device = device)
    valid_iterator = data.BucketIterator(valid_data,
                                         shuffle = True,
                                         batch_size = batch_size, 
                                         device = device)
    test_iterator = data.BucketIterator(test_data,
                                         shuffle = True,
                                         batch_size = batch_size, 
                                         device = device)
    # # Create the CNN model
    # INPUT_DIM = len(TEXT.vocab)
    # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]   
    # model = CNN(INPUT_DIM, embedding_dim, n_filters, filter_sizes, 
    #             output_dim, dropout, pad_idx = PAD_IDX)
    
    # pretrained_embeddings = TEXT.vocab.vectors
    # model.embedding.weight.data.copy_(pretrained_embeddings)
    
    # # Convert unknown token to zeros tensors
    # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    # model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    # model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
    
    
    return train_iterator, valid_iterator, test_iterator
    