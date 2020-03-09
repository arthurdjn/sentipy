# -*- coding: utf-8 -*-
# Created on Sun Feb 23 16:02:31 2020
# @author: arthurd

import os
import pandas as pd
import torch
from torchtext import data


class SSB(data.Dataset):
    """
    Stanford Sentiment Binary dataset.

    Reads in the train and dev sets of the dataset.
    Preprocesses fields where necessary.
    Builds the vocabularies for the predictor variable and label.
    Stores the datasets so it is possible to switch between them.
    """
    

    urls = ['https://github.com/arthurdjn/sentipy/raw/master/data/stanford_sentiment_binary_train.tsv.gz',
            'https://github.com/arthurdjn/sentipy/raw/master/data/stanford_sentiment_binary_dev.tsv.gz']
    name = 'ssb'
    dirname = ''
    
    
    @staticmethod
    def sort_key(data):
        # Sort data from one to another regarding the length of their text
        return len(data.text)


    def __init__(self, path, text_field, label_field, lemmatized_field = data.RawField(), **kwargs):
        
        # Get the Standford dataset fields
        SENT_ID = data.RawField()
        PHRASE_ID = data.RawField()
        
        fields = [("sent_id", SENT_ID),
                  ("phrase_id", PHRASE_ID),
                  ("label", label_field),
                  ("text", text_field),
                  ("text_pos", lemmatized_field)]

        # Create the torchtext dataset for all examples
        examples = []
        for n, entry in pd.read_table(path).iterrows():
            example = data.Example.fromlist(entry, fields)
            examples.append(example)
                    
        super(SSB, self).__init__(examples, fields, **kwargs)
                                
        
    # -------------------------------------------------------------------------
    # Class Methods
   
    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='stanford_sentiment_binary_train.tsv.gz', 
               test='stanford_sentiment_binary_dev.tsv.gz', **kwargs):
        """
        Split SSB dataset in training and testing parts.

        Parameters
        ----------
        text_field : torchtext.data.field
            The field that will be used for the sentence.
        label_field : torchtext.data.field
            The field that will be used for label data.
        root : str, optional
            Root dataset storage directory. Default is '.data'.
            The default is 'data'.
        train : str, optional
            The file's name that contains the training examples. 
            The default is 'stanford_sentiment_binary_train.tsv.gz'.
        test : str, optional
            The file's name that contains the test examples.
            The default is 'stanford_sentiment_binary_dev.tsv.gz'.
        **kwargs : Remaining keyword arguments
            Passed to the splits method of Dataset.

        Returns
        -------
        train_dataset : torchtext.dataset.Dataset
            The train dataset.
        test_dataset : torchtext.dataset.Dataset
            The test dataset.
        """
        
        path_train = root + train
        path_test = root + test
        
        if not os.path.exists(root):
            os.mkdir(root)
        
        if not os.path.exists(path_train) or not os.path.exists(path_test):
            path = cls.download(root)
            path_train = path + os.sep + train
            path_test = path + os.sep + test
        
        train_dataset = SSB(path_train, text_field, label_field, **kwargs)
        test_dataset = SSB(path_test, text_field, label_field, **kwargs)
        return train_dataset, test_dataset
        

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """
        Create iterator objects for splits of the SSB dataset.

        Parameters
        ----------
        batch_size : int, optional
            Batch size. The default is 32.
        device : int, optional
            Device to create batches on. Use - 1 for CPU and None for
            the currently active GPU device. The default is 0.
        root : str, optional
            The root directory that contains the SSB dataset subdirectory. 
            The default is '.data'.
        vectors : TYPE, optional
            One of the available pretrained vectors or a list with each
            element one of the available pretrained vectors (see Vocab.load_vectors). 
            The default is None.
        **kwargs : Remaining keyword arguments
            Passed to the splits method.

        Returns
        -------
        tuple(torchtext.data.iterator.BucketIterator)
            The iterators of the datasets.
        """        
        
        TEXT = data.Field()
        LABEL = data.LabelField(dtype = torch.float)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits((train, test), 
                                          batch_size=batch_size, 
                                          device=device)


