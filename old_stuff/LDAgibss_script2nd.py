# -*- coding: utf-8 -*-
"""
LDA Analysys on the scripts. To do the same for the plot change the dictionary 
from movie_script to movie_plot
This is on the entire dataset, we should implement the genres subset in each model
@author: macbookair
"""
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
import lda
movie_plot = {}
movie_script = {}
movie_plotvocab = {}
#vocab = []
data_compl = open ("outputfinalscoregenre.txt", "r")
for line in data_compl:
    values = line.split (" ++$++ ") 
    movie_plot[values[0]] =values[3]  #dictionary title-plot
    #vocab += values[4].split(" ")
    #all the words, with repetition
    movie_script[values[0]]=values[4]   #dicionary title-script
data_compl.close()
corpus = movie_script.values()
X = vectorizer.fit_transform(corpus) #document-term matrix
#def f7(seq):
 #   seen = set()
  #  seen_add = seen.add
   # return [ x for x in seq if not (x in seen or seen_add(x))]
#vocab = f7(vocab) 
#from collections import OrderedDict 
#vocab = list(OrderedDict.fromkeys(vocab)) #eliminate double words in vocab
#this vocabulary probably has to be fixed. the number of words is different from 
#the one that is created automatically. How to access that automatic vocabulary?
#print "vocab is "

vocabdic = vectorizer.vocabulary_ #extract vocabulary
vocab= vocabdic.keys()
print len(vocab)
print vocab
#print vocab
#print X.shape
titles = movie_script.keys()
#print titles
model = lda.LDA(n_topics=20, n_iter=200, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#NOW WE COMPUTE THE The document-topic distributions
doc_topic = model.doc_topic_
for i in range(550):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))