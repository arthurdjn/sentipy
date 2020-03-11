# -*- coding: utf-8 -*-
# Created on Tue Mar 10 12:26:13 2020
# @author: arthurd


# PyTorch
import torch

# Data science
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import time

# SentiPy modules
from sentipy.utils import get_time, one_hot



class Performer:
    
    def __init__(self, model = None, criterion = None, optimizer = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Performances
        self.reset()
        
    
    def reset(self):
        self.results_train = {"loss": None,
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "macro_f1": None,
                            "confusion_matrix": None}
        self.results_eval = {"loss": None,
                            "accuracy": None,
                            "precision": None,
                            "recall": None,
                            "macro_f1": None,
                            "confusion_matrix": None}
        self.results_test = {"loss": None,
                             "accuracy": None,
                             "precision": None,
                             "recall": None,
                             "macro_f1": None,
                             "confusion_matrix": None}
        
        self.performance = {"train": {"loss": [],
                                      "accuracy": [],
                                      "precision": [],
                                      "recall": [],
                                      "macro_f1": [],
                                      "confusion_matrix": []},
                            "eval":  {"loss": [],
                                      "accuracy": [],
                                      "precision": [],
                                      "recall": [],
                                      "macro_f1": [],
                                      "confusion_matrix": []}}
    
        
    def get_accuracy(self, y_tilde, y):
        
        assert y_tilde.shape == y.shape, ("predicted classes and gold labels should have the same shape")
        
        correct = (y_tilde == y).float() # convert into float for division 
        acc = correct.sum() / len(correct)
        
        return acc
    
    
    def get_metrics(self, gold_labels, predictions):
        metrics = {"precision": precision_score(predictions, gold_labels, average='macro'),
                   "recall":    recall_score(predictions, gold_labels, average='macro'),
                   "macro_f1":  f1_score(predictions, gold_labels, average='macro'),
                   "confusion_matrix": confusion_matrix(predictions, gold_labels).tolist()}
        return metrics
    
    def _update_performance(self):
        for (key, value) in self.results_train.items():
            self.performance["train"][key].append(value)
        for (key, value) in self.results_eval.items():
            self.performance["eval"][key].append(value)
    
    
    def train(self, iterator, log=True):
        
        # Initialize the variables
        start_time = time.time()
        epoch_loss = 0
        epoch_acc  = 0
        predictions = torch.tensor([], dtype=torch.long)
        gold_labels = torch.tensor([], dtype=torch.long)
        
        # Train mode
        self.model.train()
        for (idx, batch) in enumerate(iterator):
            self.optimizer.zero_grad()
            
            y_hat = self.model(batch.text)
            # Get the predicted classes
            # y_tilde = self.model.predict(y_hat)
            y_tilde = torch.argmax(y_hat, dim = 1)

            predictions = torch.cat((predictions, y_tilde))
            gold_labels = torch.cat((gold_labels, batch.label.long()))
    
            loss = self.criterion(y_hat, batch.label.long())
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            acc = self.get_accuracy(y_tilde, batch.label)   
            epoch_acc += acc.item()
            
            if log:
                percentage = idx * 100 // (len(iterator) - 1)
                loading = "=" * (percentage // 2) + " " * (50 - percentage//2)
                time_min, time_sec = get_time(start_time, time.time())
                print("\rTraining  :   {0:3d}% | [{1}] | Time : {2}m {3}s".format(percentage, loading, time_min, time_sec), end="")
            
        if log:
            print()
        
        loss = epoch_loss / len(iterator)
        accuracy = epoch_acc / len(iterator)
        
        # Update the performance
        metrics = self.get_metrics(gold_labels, predictions)
        self.results_train["loss"] = loss
        self.results_train["accuracy"] = accuracy
        self.results_train["precision"] = metrics["precision"]
        self.results_train["recall"] = metrics["recall"]
        self.results_train["macro_f1"] = metrics["macro_f1"]
        self.results_train["confusion_matrix"] = metrics["confusion_matrix"]
        
        return loss, accuracy



    def evaluate(self, iterator, log=True):
        
        # Initialize the variables
        start_time = time.time()
        epoch_loss = 0
        epoch_acc  = 0
        predictions = torch.tensor([], dtype=torch.long)
        gold_labels = torch.tensor([], dtype=torch.long)
        
        # Eval mode        
        self.model.eval()
        with torch.no_grad():
            
            for (idx, batch) in enumerate(iterator):               
                y_hat = self.model(batch.text)
                # Get the predicted classes
                y_tilde = torch.argmax(y_hat, dim = 1)
                predictions = torch.cat((predictions, y_tilde))
                gold_labels = torch.cat((gold_labels, batch.label.long()))
        
                loss = self.criterion(y_hat, batch.label.long())
                epoch_loss += loss.item()
                acc = self.get_accuracy(y_tilde, batch.label)   
                epoch_acc += acc.item()
                
                if log:
                    percentage = idx * 100 // (len(iterator) - 1)
                    loading = "=" * (percentage // 2) + " " * (50 - percentage//2)
                    time_min, time_sec = get_time(start_time, time.time())
                    print("\rValidation:   {0:3d}% | [{1}] | Time : {2}m {3}s".format(percentage, loading, time_min, time_sec), end="")
                
        if log:
            print()
                
        loss = epoch_loss / len(iterator)
        accuracy = epoch_acc / len(iterator)
        
        # Update the performance
        metrics = self.get_metrics(gold_labels, predictions)
        self.results_eval["loss"] = loss
        self.results_eval["accuracy"] = accuracy
        self.results_eval["precision"] = metrics["precision"]
        self.results_eval["recall"] = metrics["recall"]
        self.results_eval["macro_f1"] = metrics["macro_f1"]
        self.results_eval["confusion_matrix"] = metrics["confusion_matrix"]
        
        return loss, accuracy
    
    
    def test(self, iterator, thresholds = (0.5, 0.5), addneutral=False, log = True):
        
        # Initialize the variables
        start_time = time.time()
        epoch_loss = 0
        epoch_acc  = 0
        predictions = torch.tensor([], dtype=torch.long)
        gold_labels = torch.tensor([], dtype=torch.long)
        
        # Eval mode        
        self.model.eval()
        with torch.no_grad():
            
            for (idx, batch) in enumerate(iterator):               
                y_hat = self.model(batch.text)
                # Get the predicted classes
                if addneutral:
                    y_tilde = self.model.predict(y_hat, thresholds)
                    batch.label = batch.label[y_tilde != 2]
                    y_tilde = y_tilde[y_tilde != 2]
                    y_hat = one_hot(y_tilde, 2)
                else:
                    y_tilde = torch.argmax(y_hat, dim=1)
                    
                predictions = torch.cat((predictions, y_tilde))
                gold_labels = torch.cat((gold_labels, batch.label.long()))
                
                loss = self.criterion(y_hat, batch.label.long())
                epoch_loss += loss.item()
                acc = self.get_accuracy(y_tilde, batch.label)   
                epoch_acc += acc.item()
                
                if log:
                    percentage = idx * 100 // (len(iterator) - 1)
                    loading = "=" * (percentage // 2) + " " * (50 - percentage//2)
                    time_min, time_sec = get_time(start_time, time.time())
                    print("\rTest     :   {0:3d}% | [{1}] | Time : {2}m {3}s".format(percentage, loading, time_min, time_sec), end="")
                
        if log:
            print()
                
        loss = epoch_loss / len(iterator)
        accuracy = epoch_acc / len(iterator)
        
        # Update the performance
        metrics = self.get_metrics(gold_labels, predictions)
        self.results_test["loss"] = loss
        self.results_test["accuracy"] = accuracy
        self.results_test["precision"] = metrics["precision"]
        self.results_test["recall"] = metrics["recall"]
        self.results_test["macro_f1"] = metrics["macro_f1"]
        self.results_test["confusion_matrix"] = metrics["confusion_matrix"]
        
        return accuracy
    
    

    
    def run(self, train_iterator, valid_iterator = None,
            epochs = 10, log = True):
    
        for epoch in range(epochs):
            if log:
                print("Epoch     : {0:3d}/{1}".format(epoch + 1, epochs))
            
            self.train(train_iterator, log=log)
            if valid_iterator is not None:
                self.evaluate(valid_iterator, log=log) 
            
            self._update_performance()
            
            if log: 
                print("Stats Training     | Loss: {0:.3f} | Acc: {1:.2f}%".format(self.performance["train"]["loss"][-1], 
                                                                                  self.performance["train"]["accuracy"][-1]*100), 
                      end=" | ")
                print("Prec.: {0:.2f}% | Rec.: {1:.2f}% | F1: {0:.2f}%".format(self.performance["train"]["precision"][-1]*100, 
                                                                              self.performance["train"]["recall"][-1]*100, 
                                                                              self.performance["train"]["macro_f1"][-1]*100))
                print("Stats Validation   | Loss: {0:.3f} | Acc: {1:.2f}%".format(self.performance["eval"]["loss"][-1], 
                                                                                  self.performance["eval"]["accuracy"][-1]*100), 
                      end=" | ")
                print("Prec.: {0:.2f}% | Rec.: {1:.2f}% | F1: {0:.2f}%\n".format(self.performance["eval"]["precision"][-1]*100, 
                                                                              self.performance["eval"]["recall"][-1]*100, 
                                                                              self.performance["eval"]["macro_f1"][-1]*100))  
    

        
    