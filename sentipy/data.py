# -*- coding: utf-8 -*-
# Created on Tue Mar 10 14:55:29 2020
# @author: arthurd


import torch
from torchtext import data
import torch.nn.functional as F

# SentiPy modules
from sentipy.model.model import CNN
from sentipy.dataset.sstb import SSTB
from sentipy.utils import tokenizer


class Data:
    
    def __init__(self, text_field, label_field, 
                 train_data = None, eval_data = None, test_data = None):
        
        # Fields
        self.TEXT = text_field
        self.LABEL = label_field
        # Data
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

    
    def get_dataset(self, dataset, split = 0.8, **kwargs):
        data = dataset.splits(self.TEXT, self.LABEL, **kwargs)
        
        if len(data) == 2:
            train_data, eval_data = data[0].split(split)
            self.train_data = train_data
            self.eval_data = eval_data
            self.test_data = data[1]
        else:
            self.train_data = data[0]
            self.eval_data = data[1]
            self.test_data = data[2]
            

    def process(self):
        pass
        
        
        
        
        
        
        
        
        
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    
    # # Pre processing the Text / Labels
    # TEXT = data.Field(tokenize = tokenize, batch_first = True)   # you can add "tokenize = 'spacy'" 
    # LABEL = data.LabelField(dtype = torch.float)
    
    # # 1/ Get the training data / validation data and test data
    # train_data, test_data = SSB.splits(TEXT, LABEL)
    # train_data, valid_data = train_data.split(0.8)
    
    # # 2/ Create the vocabulary for words embeddings
    # TEXT.build_vocab(train_data, 
    #                  max_size = vocab_size, 
    #                  vectors = "glove.6B.100d", 
    #                  unk_init = torch.Tensor.normal_)
    # LABEL.build_vocab(train_data)
    
    # # 3/ Create iterators
    # train_iterator = data.BucketIterator(train_data, 
    #                                      shuffle = True,
    #                                      batch_size = batch_size, 
    #                                      device = device)
    # valid_iterator = data.BucketIterator(valid_data,
    #                                      shuffle = True,
    #                                      batch_size = batch_size, 
    #                                      device = device)
    # test_iterator = data.BucketIterator(test_data,
    #                                      shuffle = True,
    #                                      batch_size = batch_size, 
    #                                      device = device)
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