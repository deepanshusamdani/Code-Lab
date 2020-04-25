#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:18:58 2020

@author: deepu
"""

from nltk.corpus import wordnet
import webbrowser
from termcolor import colored  #syntax
#print((colored(text, color, on_color, attrs)), **kwargs)

#function to define the definition and example of the word
def def_example(word):
    w = "Word:",word
    print(colored(w,'white','on_grey',['bold']))
    print("\n")
    name,def_,pos = "","",""
    example=[]
    
    try:
        syn = wordnet.synsets(word)[0]
        name = syn.name()
        def_ = syn.definition()
        example = syn.examples()
        pos =syn.pos() #partsOfSpeech
    except IndexError:
        pass
    return name,def_,example,pos

#function to find the Synonyms & Antonyms of word 
def synonym_antonym(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return synonyms, antonyms

#function to find the Abstract & Specific words
def hyper_hypo(word):
    abstractTerm = []  #hypernyms
    specificTerm = []  #hyponyms
    
    try:        
        syn = wordnet.synsets(word)[0]        
        hyper = syn.hypernyms()
        #to check whether the list of hypernyms is empty
        if len(hyper) == 0:
            print(colored("Their Is Not Any Abstract Or Specific Word \n",'red',None,['dark']))
            return abstractTerm, specificTerm
        
        else:          #len(hyper) !=0
            b = hyper[0].lemmas()
            
            for i in range(len(b)):
                c  = (b[i]).name()
                abstractTerm.append(c)
    
            k = hyper[0].hyponyms()
            specificTerm.append(k)
            return abstractTerm, specificTerm
    
    except IndexError:
        pass
    
    return abstractTerm, specificTerm

#Google Sarche for entered word
def googleSearch(word):
    search_terms = [word]
    for term in search_terms:
        #url = "https://en.wikipedia.org/wiki/Special:Search?search=".format(term)
        url = "https://www.google.com.tr/search?q={}".format(term)
        gsr = webbrowser.open_new_tab(url)     
    return gsr

#play song for entered word
def youTube(word):
    url = "https://www.youtube.com/results?search_query".format(word)
    uTube = webbrowser.open_new_tab(url)
    return uTube
    
def main():
    #enter the number for how many words to enterd
    print(colored("Enter the number: ",'red'))
    num = int(input())
    wordList = []
    
    #enter the word 
    for i  in range(num):
        print(colored("Enter the word equals to number %d: \n",'red')%(i+1))
        word = input()
        wordList.append(word)
    
    #iterate over the word
    for i in range(len(wordList)):
         #call the function for basic definition
         name,def_,example,pos = def_example(wordList[i])
         if(len(name)!=0):
             #print((colored(text, color, on_color, attrs)), **kwargs)
             #print(colored("name: ",'green',None,['blink','underline,'dark','bold']),word)
             print(colored("Name:",'blue'),name,"\n")
             print(colored("Definition: ",'blue'),def_,"\n")
             print(colored("Example: ",'blue'),example,"\n")
             print(colored("PartsOfSpeech: ",'blue'),pos,"\n")
         
         #call the function to Synonym and Anotnym
         syno,anto = synonym_antonym(wordList[i])
         uniqSyn = set(syno)
         uniqAnt = set(anto)
         #print("\n")
         print(colored("Synonym: ",'blue'),uniqSyn,"\n")
         print(colored("Antonym: ",'blue'),uniqAnt,"\n")
        
         #call the function of Abstract and specific words
         abstract, specific = hyper_hypo(wordList[i])
         if ((len(abstract) !=0) and (len(specific)!= 0)):
             print(colored("AbstractWord: ",'blue'),abstract,"\n")
             print(colored("SpecificWord: ",'blue'),specific,"\n")
         else:
             print(colored("AbstractWord: ",'blue'),abstract,"\n")
             print(colored("SpecificWord: ",'blue'),specific,"\n")

         #call the function googleSearch more details of that word
         gserch = googleSearch(wordList[i])
         print(colored("Google Page: ",'blue'),gserch,"\n")
         
         #call function youtube from word
         youT = youTube(wordList[i])
         print(colored("youTube: ",'blue'),youT,"\n")
    
if __name__ == "__main__":
    main()
    
    


  
#print ("\nSynset root hypernerm :  ", syn.root_hypernyms()) 
