#!/usr/bin/python3

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.summarizer import summarize
#import string 


#NOTE::--> input sentance must be greater than one sentence the only be text can summarize. 
#Removing stop words and making frequency table
text=("Yoga is a good practice if one does in daily life. + \
 It helps to live healthy  life style and better life forever. + \
 We should let our kids know about the benefits of yoga as well as practice yoga in daily routine. + \
 Yoga Essay is a general topic which students get in the schools during essay writing. + \
 Enhance your kids essay writing skills by using such type of simply written essay on yoga and its benefits. + \
 This Yoga Essay will help your child’s health and English skill as well.")
#print(summarize(text))

print("len of text : " ,len(text))
stopwords=set(stopwords.words("english"))
words=word_tokenize(text)
print("words : " ,words)

# sent=sent_tokenize(text)
# print("sent : " , sent)


join_words=' '.join(words)
print("summary of words : " , summarize(join_words))


#creating a frequency table
freqTable=dict()

for word in words:
	word=word.lower()
	# rmv_punc=[i for i in word if i not in string.punctuation]

	# text_punc_clean=''.join(rmv_punc)
	# # print(text_punc_clean)
	# text_token=word_tokenize(text_punc_clean)
	# text_token_join=''.join(text_token)
	# print(text_token_join)
	if word in stopwords:                     #stopwords
		continue
	if word in freqTable:
		freqTable[word] = freqTable[word]+1
		print(word , freqTable[word])
	else:
		freqTable[word] = 1


#assigning a score to every sentence
sentences=sent_tokenize(text)
sentenceValue=dict()


#print("sentences : " ,sentences)
#print(sentenceValue)

for sentence in sentences:
    for wordValue in freqTable:
    	print(wordValue)
    	if wordValue[0] in sentence.lower():		#Index 0 of wordValue will return the word itself and 1 return the no. of instances
    		if sentence[:4] in sentenceValue:
    			sentenceValue[sentence[:4]] += wordValue[1]
    			
    		else:
    			sentenceValue[sentence[:4]] = wordValue[1]


# sumValues = 0
# for sentence in sentenceValue:
#     sumValues += sentenceValue[sentence]

# # Average value of a sentence from original text
# average = int(sumValues/ len(sentenceValue))
# print(average)