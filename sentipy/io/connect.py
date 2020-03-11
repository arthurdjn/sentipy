# -*- coding: utf-8 -*-
# Created on Sun Mar  8 23:39:45 2020
# @author: arthurd


import configparser
import tweepy



def connect(loginfile):
    """
    Read the login.ini file and get the tokens to connect on Twitter API.

    Parameters
    ----------
    loginfile : str
        Path to the login.ini file.

    Returns
    -------
    login : dict
        - consumer_key
        - consumer_secret
        - access_token
        - access_token_secret
    """
    # Red the ini file
    config = configparser.ConfigParser()
    config.read(loginfile)
    
    # Get the logins
    try:
        consumer_key = eval(config.get('Consumer', 'consumer_key'))
        consumer_secret = eval(config.get('Consumer', 'consumer_secret'))
        access_token = eval(config.get('Access', 'access_token'))
        access_token_secret = eval(config.get('Access', 'access_token_secret'))
    except ValueError as e:
        raise e("one or more of te API key fields are empty")
        
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
        
    return auth






# import tweepy

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)

# api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)


    

