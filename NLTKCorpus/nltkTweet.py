#!/usr/bin/python3

import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

#creating consumer keys
consumer_key='LY7g0NUTBbRqimm5XAFWIUUbE'
consumer_secret= 'iXSSyImidCWIiclvwh9hC6bPkkF1YjYpNKNC93RMQmYUc6ECpg'
#creating access keys
access_key='774600536968527872-Ee5SrpNF29e4hBg5sDrN7Ft6R0XQIkM' 


access_secret='qkxiGFun50SVKEgcpyBE8aubkL0IAHlxy48nT7VLRdrse'

#connection for auth
#here auth is very musch simillar to session variable

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token (access_key,access_secret)

#connecting API 
connect=tweepy.API(auth)

#now finding data
get_data=connect.search('Cognizant')

#a=[]
#printed data
stop_words = set(stopwords.words('english'))

for i in get_data:
	print(i.text)

	# analysis=TextBlob(i.text)
	# print(analysis)
	# a=(analysis.sentiment.polarity)
	# # print(a)




	# if analysis.sentiment.polarity == 0:
	# 	print("netural") 

	# if analysis.sentiment.polarity >0:
	# 	print("positive")
	
	# if analysis.sentiment.polarity < 0:
	# 	print("negative")


