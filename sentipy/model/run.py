# -*- coding: utf-8 -*-
# Created on Fri Feb 21 15:56:42 2020
# @author: arthurd & sigurdh


# PyTorch
import torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data science
from sklearn.metrics import f1_score, recall_score, precision_score
import random
import numpy as np

# SentiPy modules
from sentipy.model.model import CNN
from sentipy.dataset.ssb import SSB
from sentipy.model.functional import evaluate, train, confusion_matrix
from sentipy.utils import tokenizer



def run(model, criterion, optimizer,
        epochs = 10,
        log = True):    
    
    # Save the results
    results = {"train": {"precision": None,
                         "recall": None,
                         "macro-f1": None,
                         "confusion_matrix": None},
               "valid": {"precision": None,
                         "recall": None,
                         "macro-f1": None,
                         "confusion_matrix": None}}
    
    for epoch in range(epochs):
        if log:
            print("Epoch     : {0:3d}/{1}".format(epoch+1, epochs))
        train_loss, train_acc, train_predictions, train_labels = train(model, train_iterator, optimizer, criterion, log=log)
        valid_loss, valid_acc, valid_predictions, valid_labels = evaluate(model, valid_iterator, criterion, log=log)
        
        # Getting the metrics
        # Training
        train_precision = precision_score(train_labels, train_predictions, average='macro')
        train_recall = recall_score(train_labels, train_predictions, average='macro')
        train_macro_f1 = f1_score(train_labels, train_predictions, average='macro')
        train_confusion = confusion_matrix(train_labels, train_predictions)        
        # Validation
        valid_precision = precision_score(valid_labels, valid_predictions, average='macro')
        valid_recall = recall_score(valid_labels, valid_predictions, average='macro')
        valid_macro_f1 = f1_score(valid_labels, valid_predictions, average='macro')
        valid_confusion = confusion_matrix(valid_labels, valid_predictions)        
        
        if log: 
            print("Stats Training     | Loss: {0:.3f} | Acc: {1:.2f}%".format(train_loss, train_acc*100), end=" | ")
            print("Prec.: {0:.2f}% | Rec.: {1:.2f}% | F1 {0:.2f}%".format(train_precision*100, train_recall*100, train_macro_f1*100))
            print("Stats Validation   | Loss: {0:.3f} | Acc: {1:.2f}%".format(valid_loss, valid_acc*100), end=" | ")
            print("Prec.: {0:.2f}% | Rec.: {1:.2f}% | F1 {0:.2f}%".format(valid_precision*100, valid_recall*100, valid_macro_f1*100))
            print()
            
        # Add the results
        # Train
        results["train"]["precision"] = train_precision
        results["train"]["recall"] = train_recall
        results["train"]["macro_f1"] = train_macro_f1
        results["train"]["confusion_matrix"] = train_confusion.tolist()
        # Validation
        results["valid"]["precision"] = valid_precision
        results["valid"]["recall"] = valid_recall
        results["valid"]["macro_f1"] = valid_macro_f1
        results["valid"]["confusion_matrix"] = valid_confusion.tolist()
    
    return results



if __name__ == "__main__":
    
    
    # Let's fix the seed
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    EPOCHS = 2
    BATCH_SIZE = 64
    VOCAB_SIZE = 20000
    EMBEDDING_DIM = 100
    N_FITLERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 2
    DROPOUT = 0.5
    ACTIVATION_LAYER = F.relu
    LR = 0.1
    WEIGHT_DECAY = 0.001
    
    
    # Pre processing the Text / Labels
    # --------------------------------
    TEXT = data.Field(batch_first = True)   # you can add "tokenize = 'spacy'" 
    LABEL = data.LabelField(dtype = torch.float)
    
    # 1/ Get the training data / validation data and test data
    train_data, test_data = SSB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(0.8)
    
    # 2/ Create the vocabulary for words embeddings
    TEXT.build_vocab(train_data, 
                     max_size = VOCAB_SIZE, 
                     vectors = "glove.6B.100d", 
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    
    # 3/ Get iterators... to iterate
    # @NOTE: The SSB dataset inherits from torchtext.dataset.Dataset class
    #        You can directly use torchtext methods on SSB, like splits / iterators...
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                (train_data, valid_data, test_data), 
                                 batch_size = BATCH_SIZE, 
                                 device = device)
    
    # Model
    # -----
    INPUT_DIM = len(TEXT.vocab)
    
    # 1/ Padding
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]   
    # Create the CNN
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FITLERS, FILTER_SIZES, 
                OUTPUT_DIM, DROPOUT, pad_idx = PAD_IDX)
    
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    
    # Convert unknown token to zeros tensors
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    
    # Optimization
    # ------------
    optimizer = optim.Adadelta(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    
    
    
    PARAMS = {"epochs": EPOCHS,
              "log": True}
    
    results = run(model, criterion, optimizer, **PARAMS)
