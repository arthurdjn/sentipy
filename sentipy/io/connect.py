# -*- coding: utf-8 -*-
# Created on Sun Mar  8 23:39:45 2020
# @author: arthurd


import tweepy





login_file = "login.ini"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


    
    
