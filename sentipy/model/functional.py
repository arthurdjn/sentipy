# -*- coding: utf-8 -*-
# Created on Fri Feb 21 15:57:03 2020
# @author: arthurd


import time
import torch
# SentiPy modules
from sentipy.utils import to_one_hot


def get_accuracy(predicted_class, y):
    correct = (predicted_class == y).float() # convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def get_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, log=True):
    
    start_time = time.time()
    labels = torch.tensor([])
    predicted = torch.tensor([])
    epoch_loss = 0
    epoch_acc = 0
    num_class = 2
    
    model.train()
    N = len(iterator)
    for (idx, batch) in enumerate(iterator):
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        predicted_classes = torch.argmax(predictions, dim = 1)
        predicted = torch.cat((predicted, to_one_hot(predicted_classes, num_class)))
        labels = torch.cat((labels, to_one_hot(batch.label, num_class)))

        loss = criterion(predictions, batch.label.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc = get_accuracy(torch.argmax(predictions, dim=1), batch.label)   
        epoch_acc += acc.item()
        
        if log:
            percentage = idx * 100 // (N - 1)
            loading = "=" * (percentage // 2) + " " * (50 - percentage//2)
            time_min, time_sec = get_time(start_time, time.time())
            print("\rTraining  :   {0:3d}% | [{1}] | Time : {2}m {3}s".format(percentage, loading, time_min, time_sec), end="")
        
    if log:
        print()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), predicted, labels



def evaluate(model, iterator, criterion, log=True):
    
    start_time = time.time()
    labels = torch.tensor([])
    predicted = torch.tensor([])
    epoch_loss = 0
    epoch_acc = 0
    num_class = 2
    
    N = len(iterator)
    
    model.eval()
    with torch.no_grad():
        
        for (idx, batch) in enumerate(iterator):

            predictions = model(batch.text)
            predicted_classes = torch.argmax(predictions, dim = 1)
            predicted = torch.cat((predicted, to_one_hot(predicted_classes, num_class)))
            labels = torch.cat((labels, to_one_hot(batch.label, num_class)))
    
            loss = criterion(predictions, batch.label.long())
            epoch_loss += loss.item()
            acc = get_accuracy(torch.argmax(predictions, dim=1), batch.label)        
            epoch_acc += acc.item()
            
            if log:
                percentage = idx * 100 // (N - 1)
                loading = "=" * (percentage // 2) + " " * (50 - percentage//2)
                time_min, time_sec = get_time(start_time, time.time())
                print("\rValidation:   {0:3d}% | [{1}] | Time : {2}m {3}s".format(percentage, loading, time_min, time_sec), end="")
            
    if log:
        print()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator), predicted, labels
       
        
def confusion_matrix(predictions, labels):
    
    num_class = predictions.shape[1]
    predictions = torch.argmax(predictions, dim=1)
    labels = torch.argmax(labels, dim=1)
    # Initialize the confusion matrix
    confusion_matrix = torch.zeros(num_class, num_class)
    
    # Iterating over all batches (can be 1 batch as well):
    for i in range(len(predictions)):
        # Update the confusion matrix
        confusion_matrix[predictions[i].long(), labels[i].long()] += 1

    return confusion_matrix



def analyze_confusion_matrix(confusion_matrix):
    n_classes = len(confusion_matrix) 
    # True positive : correct prediction, ie the diagonal of the confusion matrix
    TP = confusion_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
        FP = confusion_matrix[c, idx].sum()
        FN = confusion_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c] + FN))
        specificity = (TN / (TN + FP))
        
        # Display the analysis in the console
        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, TP[c], TN, FP, FN))
        print("Sensitivity :", sensitivity)
        print("Specificity : {0}\n------".format(specificity))













