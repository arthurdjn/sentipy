# -*- coding: utf-8 -*-
# Created on Thu Mar  5 16:05:00 2020
# @author: arthurd


import configparser


if __name__ == "__main__": 
    
    # Create the config for key arguments
    config = configparser.ArgumentParser()
    config.add_argument('--vocab_size', help="How many words types to consider", action='store',
                        type=int, default=3000)
    config.add_argument('--hidden_dim', help="Size of the hidden layer(s)", action='store',
                        type=int, default=128)
    config.add_argument('--batch_size', help="Size of mini-batches", action='store', type=int,
                        default=32)
    config.add_argument('--lr', action='store', help="Learning rate", type=float, default=1e-3)
    config.add_argument('--epochs', action='store', help="Max number of epochs", type=int,
                        default=15)
    config.add_argument('--split', action='store', help="Ratio of train/dev split", type=float,
                        default=0.9)
    
    
    
    
    
    
    
    