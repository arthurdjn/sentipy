# -*- coding: utf-8 -*-
# Created on Fri Feb 21 15:56:42 2020
# @author: arthurd & sigurdh


"""
This module contains all pytorch models used to train our classifier.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    
    Attributes
    ----------
    embedding : torch.nn.Embedding
        Embedding layer of shape (vocab_size, embedding_dim).
    convs : torch.nn.ModuleList(torch.nn.Conv2d)
        List of 2D convolutional layers.
    dropout : float
        Convolutional dropout.
    linear : torch.nn.Linear
        Linear ayer of shape (number_filters * n_filters, output_dim)
    """
        
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, 
                 output_dim, dropout = 0.5, pad_idx = None, 
                 activation_layer = F.relu, activation_output = F.softmax):
        
        # Initialize the model
        super().__init__()
        
        # Dimension shortcut
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,       
                            num_layers=1)
                        
        # List of 2D convolutional layers
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        # Max dropout
        self.dropout = nn.Dropout(dropout)
        # Linear layer to predict the class
        self.linear = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # Activation functions
        self.activation_layer = activation_layer
        self.activation_output = activation_output
        
        
    def forward(self, text):
        """
        One forward step.

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """    
        # text shape: (batch size, sent len)
        
        embedded = self.embedding(text)  
        # embedded shape: (batch_size, sent len, emb dim)

        # embedded, _ = self.lstm(embedded)
        # new embedded shape: (batch_size, sent len, emb dim)
                
        embedded = embedded.unsqueeze(1)
        # embedded shape: (batch_size, 1, sent len, emb dim)
        conved = [self.activation_layer(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n shape: (batch_size, n_filters, sent len - filter_sizes[n])

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n shape: (batch_size, n_filters)
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat shape: (batch_size, n_filters * number_filters)
        
        linear = self.linear(cat)
        # linear shape: (number_filters * n_filters, output_dim)
        
        predictions = self.activation_output(linear)
        # predictions shape: (number_filters * n_filters, output_dim)

        return predictions
    
    
    def predict(self, predictions, thresholds = (0.7, 0.7)):
        
        thresholds = torch.tensor(thresholds)
        y_tilde = torch.argmax(predictions, dim=1)
        y_tilde_multiclass = torch.where(torch.max(predictions, dim=1).values > thresholds[y_tilde],
                                         y_tilde,
                                         torch.zeros_like(y_tilde) + 2)
        
        return y_tilde_multiclass
            
    
    
    
    
    
    
    # def weights_init(self, TEXT):
    #     pretrained_embeddings = TEXT.vocab.vectors
    #     self.embedding.weight.data.copy_(pretrained_embeddings)
        
    #     # Convert unknown token to zeros tensors
    #     UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    #     PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] 
    #     self.embedding.weight.data[UNK_IDX] = torch.zeros(self.embedding_dim)
    #     self.embedding.weight.data[PAD_IDX] = torch.zeros(self.embedding_dim)
    
    
    
    
    

