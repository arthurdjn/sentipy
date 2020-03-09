# -*- coding: utf-8 -*-
# Created on Fri Feb 21 16:10:08 2020
# @author: arthurd


"""
This module contains usefull functions used for Data class.
"""

import torch


def tokenizer(sample):
    new_sample = []
    for word in sample.split():
        splits = word.split("-")
        for split in splits[:-1]:
            new_sample.append(split)
            new_sample.append("-")
        new_sample.append(splits[-1])
    return new_sample



def to_one_hot(batched_tensor, num_class):
    """
    Convert a 1D batched tensor to one hot encoded tensors.

    Parameters
    ----------
    batched_tensor : torch.tensor
        Batched tensor of shape batch_size. The tensor type should be integers
        corresponding to the predicted class.
    num_class : int
        Number of classes.

    Returns
    -------
    one_hot : torch.tensor
        One hot encoded tensors.
    """
    one_hot = torch.zeros((batched_tensor.size()[0], num_class))
    one_hot[torch.arange(batched_tensor.size()[0]), batched_tensor.long()] = 1
    return one_hot