# -*- coding: utf-8 -*-
# Created on Mon Mar  9 18:46:10 2020
# @author: arthurd

from nltk.tokenize import TweetTokenizer 



def tokenizer_naive(sample):
    new_sample = []
    for word in sample.split():
        splits = word.split("-")
        for split in splits[:-1]:
            new_sample.append(split)
            new_sample.append("-")
        new_sample.append(splits[-1])
    return new_sample



def tokenizer_tweets(sample):
    # Tokenize using an existing methods
    words = TweetTokenizer().tokenize(sample)
    # Remove @username, https://gif, img, etc. and add <element> instead
    tokenized_sample = []
    for word in words:
        if word[0] == "@":
            tokenized_sample.append("<user>")
        elif word[:4] == "http":
            tokenized_sample.append("<url>")
        elif word == "$":
            pass
        # elif word[0] == "#":
        #     tokenized_sample.extend(("<hashtag>", word[1:].lower()))
        else:
            tokenized_sample.append(word.lower())
    return tokenized_sample
    
    
    
    