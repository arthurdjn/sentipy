# -*- coding: utf-8 -*-
# Created on Fri Feb 21 15:57:33 2020
# @author: arthurd

from sentipy import SentimentData



def test_SentimentData():
    dirname = "../data"
    sentiment_data = SentimentData(dirname = dirname)
    stop_words = sentiment_data.stop_words
    fields = sentiment_data.fields
    train = sentiment_data.fields
    dev = sentiment_data.dev
    dataset = sentiment_data.dataset
    tokens = sentiment_data.tokens
    labels = sentiment_data.labels
    vocab = sentiment_data.vocab
    
    print("\t1/ Size of stop words : {}".format(len(stop_words)))
    print("\t2/ Fields : {}".format(fields))
    print("\t3/ Train : {}".format(train))
    print("\t4/ Dev : {}".format(dev))
    print("\t5/ Dataset : {}".format(dataset))
    print("\t6/ Tokens : {}".format(tokens))
    print("\t7/ Labels : {}".format(labels))
    print("\t8/ Vocab size : {}".format(len(vocab)))
    
    # Changing state
    sentiment_data.state = "dev"
    print("\t9/ Setting dev state : {}".format(sentiment_data.state))
    
    # Getitem
    index = 0
    item = sentiment_data[index]
    print("\t10/ Item at index {} is : {}".format(index, item))
    print("\t11/ Tokens at index {} is : {}".format(index, item['tokens']))




if __name__ == "__main__":
    
    # Test the SentimentData class
    print("Testing SentimentData class...")
    test_SentimentData()
    print("SentimentData tested.")
    
    
    