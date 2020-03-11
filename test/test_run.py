# -*- coding: utf-8 -*-
# Created on Mon Mar  9 15:47:56 2020
# @author: arthurd


# PyTorch
import torch.nn as nn
import torch.optim as optim


from sentipy.model.run import run
from sentipy.model.process import process



def test_process():
    model = process()

def test_run():
    model = process()
    optimizer = optim.Adadelta(model.parameters(), lr = 0.1, weight_decay = 0.001)
    criterion = nn.CrossEntropyLoss()
    
    run(model, criterion, optimizer)




if __name__ == "__main__":
    
    # Pre process
    test_process()
    
    
    # Run
    test_run()