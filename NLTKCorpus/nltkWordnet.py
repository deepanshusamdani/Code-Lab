from nltk.corpus import wordnet
from nltk.tag import pos_tag
import nltk
from nltk.stem import PorterStemmer 
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank

syn=wordnet.synsets("ego")

# print(syn[1])
#print(syn[0].name())

print("name : ",syn[0].lemmas()[0].name())
# z=wordnet.synsets(y)

print("definition : ",syn[0].definition())

print("examples : ",syn[0].examples())

#difference between Lemmatizer and Stemmer is of output
#stemmer the word 
#reason behind the stemmer the word is that , suppose if we not stemmere the word then each word is consider as a field and each-
#field have one coloumn in the sparse  matrix so ,as no. of column increase , so as complexity is also increases.....
#ex:-- word= love ,, loved ,,loves (means all three are differeent column...)so stemmer the word (it become --only "love")
stemmer=PorterStemmer()
print("stemmer the word : ",stemmer.stem('increases'))

#Lemmatizing Words Using WordNet
lemmatizer = WordNetLemmatizer()
print("lemm the word : ", lemmatizer.lemmatize('increasing', pos='v'))   #v=verb n=noun a=adverb adj. 

#meaning of the word
#sent="love the life and live the happy life and love yourself. my life is great for me and i am writer of love poetry."
sent= "python is the scripting language , usen in data science . Machine learning algo should be implement using python is the best one."
#sent="i love my country. i love my udaipur."
tokens=nltk.word_tokenize(sent)
#print(tokens)
pos_tag=nltk.pos_tag(tokens)
print("pos tag : " ,pos_tag)
#draw the tree of pos_tag words
tree = treebank.parsed_sents()[0]
tree.draw()

grammar = "NP: {<NN>?<JJ>*<JJS>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sent)
print(result)

#synonyms and antonyms of the word
synonyms=[]
antonyms=[]


for syn in wordnet.synsets("happy"):
	for i in syn.lemmas():
		synonyms.append(i.name())
		
		if i.antonyms():
			antonyms.append(i.antonyms()[0].name())
synonym=(set(synonyms))
#convert dictionary to list
dict_list_synonym=list(synonym)
print("synonyms : ",dict_list_synonym)
# print(dict_list_synonym[0])
antonym=(set(antonyms))
#convert dictionary to list
dict_list_antonym=list(antonym)
print("antonyms : ",dict_list_antonym)


#print(Wup_Similarity('happy','glad'))
# com=Wup_Similarity(dict_list_synonym[0] , dict_list_antonym[0])
# compare.append(com)
#print(max(com))



#all keywords

# TAG_WORD_SET = set(["VBD", "VBG", "VBN", "VB", "JJ", "RB", "NN", "NNS", "NNP"])
# LEMMATIZED_SET = set(["VBD", "VBG", "VBN", "VB"])
# # Our simple grammar from class (and the book)
# GRAMMAR =   """
#             N: {<PRP>|<NN.*>}
#             V: {<V.*>}
#             ADJ: {<JJ.*>}
#             NP: {<DT>? <ADJ>* <N>+}
#             PP: {<IN> <NP>}
#             VP: {<TO>? <V> (<NP>|<PP>)*}
#             """
