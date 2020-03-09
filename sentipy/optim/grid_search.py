# -*- coding: utf-8 -*-
# Created on Thu Feb 27 13:04:57 2020
# @author: arthurd


# PyTorch
import torch
import torch.nn.functional as F

# Data science
from itertools import combinations
import numpy as np
import time
import json

# Sentipy
from sentipy import run



if __name__ == "__main__":
    
    # Run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Hyper parameters testing    
    epochs_range = np.arange(300, 600, 100).tolist()
    batch_range = [32, 64, 128]
    lr_range = np.linspace(0.001, 0.3, 20)
    window_depth = [1, 2, 3]
    window_dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # epochs_range = np.arange(2, 3, 100).tolist()
    # batch_range = [64]
    # lr_range = np.linspace(0.1, 0.2, 1)
    # window_depth = [3]
    # window_dimensions = [3, 4, 5]
    window_sizes = [list(combination) for depth in window_depth 
                    for combination in list(combinations(window_dimensions, depth))]    
    activation_functions = [F.relu, F.elu, F.leaky_relu, F.tanh]
    
    # Basic display of all combinations
    print("              batch_range :", batch_range)
    print("             epochs_range :", epochs_range)
    print("                 lr_range :", lr_range)
    print("             window_sizes :", window_sizes)
    print("     activation_functions :", activation_functions)
    print("Number of models to train : {0:,}\n".format( 
        len(lr_range) * len(epochs_range) * len(batch_range) * 
        len(window_sizes) * len(activation_functions)))
      
    # Run
    # ---
    best_valid_loss = float('inf')
    predictions = []  
    
    # Let's make the grid search in N-dimensions
    results = {"config": []}
    best_accuracy = 0
    best_conf = None
    
    # select the batch size
    for batch in batch_range:
        # select the epochs
        for epochs in epochs_range:
            # Select the learning rate 
            for lr in lr_range:
                # Select the window_size
                for window_size in window_sizes:
                    # Select an activation function
                    for activation_layer in activation_functions:
                        # Run the model with these parameters
                        start = time.time()
                        params = {"epochs": int(epochs),
                                  "batch_size": int(batch),
                                  "vocab_size": 20000,
                                  "embedding_dim": 100,
                                  "n_filters": 100,
                                  "filter_sizes": window_size,
                                  "output_dim": 2,
                                  "dropout": 0.5,
                                  "lr": float(lr),
                                  "activation_layer": activation_layer.__name__,
                                  "log": False}
                
                        result = run(**params)

                        # Let's add these results to a json data file, 
                        # for better plots.
                        results['config'].append({**params,
                                                "time": time.time() - start,
                                                 **result})
    
    # Save it !
    filename = 'grid_search_SSB.json'
    with open(filename, 'w') as f:
        json.dump(results, f)

    # Load it !
    with open(filename, 'r') as f:
        data = json.load(f)
        