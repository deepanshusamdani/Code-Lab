#!/usr/bin/python3

import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

Consumer_key='LY7g0NUTBbRqimm5XAFWIUUbE'

Consumer_Secret= 'iXSSyImidCWIiclvwh9hC6bPkkF1YjYpNKNC93RMQmYUc6ECpg'

Access_Token ='774600536968527872-Ee5SrpNF29e4hBg5sDrN7Ft6R0XQIkM' 
Access_Token_Secret= 'qkxiGFun50SVKEgcpyBE8aubkL0IAHlxy48nT7VLRdrse'

#connecting for authentication

auth=tweepy.OAuthHandler(Consumer_key,Consumer_Secret)
auth.set_access_token(Access_Token , Access_Token_Secret)

#connecting with the api
connect = tweepy.API(auth)

tweets = connect.search('Rishav Sharma')

senti=[]
#printing data
for tweet in tweets:
	tweet_text=tweet.text

	#removing the punctuation marks from tweet obtained
	tweet_without_punc=[i for i in tweet_text if i not in string.punctuation]
	# print("#------------------------------------------without punc--------------------------------------------------------")
	# print(tweet_without_punc)
	# print("#------------------------------------------join--------------------------------------------------------")
	

	#join the tweet after removing punctuation
	tweet_punc_clean=''.join(tweet_without_punc)
	print(tweet_punc_clean)
	
	# tokenizing words
	# print("#------------------------------------------tokenize--------------------------------------------------------")
	tweet_without_punc= word_tokenize(tweet_punc_clean)
	# print(tweet_without_punc)
	
	# removing the stopwords
	tweet_stopwords=[i for i in tweet_without_punc  if i not in stopwords.words('english')]
	tweet_stopwords=' '.join(tweet_stopwords)
	#print(tweet_stopwords)
	
	analysis=TextBlob(tweet_stopwords)
	senti.append(analysis.sentiment.polarity)
	print(senti)


neut=[]
happy=[]
sorrow=[]


# distributing the emotions according to the polarity
for polarity in senti:
	if polarity==0:
		neut.append("netural")

	if polarity>0:
		happy.append("positive")

	if polarity<0:
		sorrow.append("negative")

#calculating the length of each emotion
neutral=len(neut)
dukhi=len(sorrow)
khush=len(happy)

# plotting Bar graph
plt.bar(['khush','neutral','dukhi'],[khush,neutral,dukhi],color="red")

plt.show()