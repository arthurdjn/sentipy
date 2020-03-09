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
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
        
        predictions = self.activation_output(linear, dim = 1)
        # predictions shape: (number_filters * n_filters, output_dim)

        return predictions

