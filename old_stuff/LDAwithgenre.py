# -*- coding: utf-8 -*-
"""
@author: ioanna
"""

# -*- coding: utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
import lda
movie_plot = {}
movie_script = {}
movie_plotvocab = {}
movie_genres = {}
#vocab = []
data_compl = open ("outputfinalscoregenre.txt", "r")
for line in data_compl:
    values = line.split (" ++$++ ") 
    movie_plot[values[0]] =values[3]  #dictionary title-plot
    #vocab += values[4].split(" ")
    #all the words, with repetition
    movie_genres[values[0]]=values[2]
    movie_script[values[0]]=values[4]   #dicionary title-script
data_compl.close()
#print movie_genres
#define dictionaries within every genre
movie_thriller = {}#243
movie_drama = {} #290   thriller+drama = 105
movie_action = {} #139  thriller + action = 85
movie_scifi = {} #101
movie_crime = {} #134   thriller + crime = 97
movie_horror = {} #84
movie_mystery = {} #96
movie_romance = {} #123   drama + romance = 79
movie_adventure = {} #91
movie_comedy = {} #149    comedy + romance = 60
movie_war = {} #20
movie_animation = {} #13
movie_western = {} #11
movie_documentary = {} #1
movie_sport = {} #8
movie_fantasy = {} #63
#i=0
#for id in movie_genres.keys():
 #   if "comedy" in movie_genres[id]:
  #      if "romance" in movie_genres[id]:
  #          i +=1
#print i
#print movie_thriller

for id in movie_genres.keys():
    if "thriller" in movie_genres[id]:
        movie_thriller[id]=movie_script[id]  #dictionary thriller movie title + script
         
#thril= open ("provathriller.txt","w")
#for id in movie_thriller.keys():
#    thril.write(id + " ++$++ " + movie_thriller[id])
#thril.close()
corpus = movie_thriller.values()

X = vectorizer.fit_transform(corpus) #document-term matrix
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))

vocabdic = vectorizer.vocabulary_ #extract vocabulary
vocab= vocabdic.keys()
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))

titles = movie_thriller.keys()
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))
#print titles

model = lda.LDA(n_topics=20, n_iter=1000)#, random_state=1)
model.fit_transform(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works

#top words for each document
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
#NOW WE COMPUTE THE The document-topic distributions
doc_topic = model.doc_topic_
for i in range(243):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
   



