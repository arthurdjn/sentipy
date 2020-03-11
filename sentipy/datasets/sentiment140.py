# -*- coding: utf-8 -*-
# Created on Sun Feb 23 16:02:31 2020
# @author: arthurd

import os
import pandas as pd
import torch
from torchtext import data


class Sentiment140(data.Dataset):
    """
    Twitter Sentiment140 dataset.

    Reads in the train and dev sets of the dataset.
    Preprocesses fields where necessary.
    Builds the vocabularies for the predictor variable and label.
    Stores the datasets so it is possible to switch between them.
    """
    

    urls = ["http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"]
    name = 's140'
    dirname = ''
    
    
    @staticmethod
    def sort_key(data):
        # Sort data from one to another regarding the length of their text
        return len(data.text)


    def __init__(self, path, text_field, label_field, 
                 keepneutral = False, neutral = None,
                 size = None, shuffle = True, **kwargs):
        
        # Get the Standford dataset fields
        SENT_ID = data.RawField()
        DATE = data.RawField()
        QUERY = data.RawField()
        USER = data.RawField()
        fields = [("label", label_field),
                  ("id", SENT_ID),
                  ("date", DATE),
                  ("query", QUERY),
                  ("user", USER),
                  ("text", text_field)]

        # Create the torchtext dataset for all examples
        examples = []
        df = pd.read_csv(path, encoding='latin-1', header=0,
                    names=["label", "id", "date", "query", "user", "text"])
                
        if shuffle:
            df = df.sample(frac=1)
            
        if neutral is not None:
            df_neutral = pd.read_csv(neutral, index_col=0, header=None).T
            df_neutral.columns = ["text"]
            df_neutral["label"] = [2] * len(df_neutral)
            df = pd.concat([df_neutral, df])
            
        for (_, entry) in df.iloc[0:size].iterrows():
            if not keepneutral and entry["label"] == 2:
                continue
            example = data.Example.fromlist(entry, fields)
            examples.append(example)
                        
        super(Sentiment140, self).__init__(examples, fields, **kwargs)
                                        
        
    # -------------------------------------------------------------------------
    # Class Methods
   
    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='training.1600000.processed.noemoticon.csv', 
               test='testdata.manual.2009.06.14.csv', 
               neutral = None, **kwargs):
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
            The default is 'training.1600000.processed.noemoticon.csv'.
        test : str, optional
            The file's name that contains the test examples.
            The default is 'testdata.manual.2009.06.14.csv'.
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
            path_train = path + train
            path_test = path + test
        
        train_dataset = Sentiment140(path_train, text_field, label_field, neutral=neutral, **kwargs)
        test_dataset = Sentiment140(path_test, text_field, label_field, **kwargs)
        
        return train_dataset, test_dataset
        

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """
        Create iterator objects for splits of the Sentiment140 dataset.

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


