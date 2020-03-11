# -*- coding: utf-8 -*-
# Created on Mon Mar  9 15:48:22 2020
# @author: arthurd

import tweepy
from sentipy.io.connect import connect


LOGIN_FILE = "../login.ini"


def test_connect():
    auth = connect(LOGIN_FILE)
    print("AUTH :", auth)
    
    
    
if __name__ == "__main__":
    
    # Connect to twitter
    test_connect()
    
    # Read tweets
    
    
    
    