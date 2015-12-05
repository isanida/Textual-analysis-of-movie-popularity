from __future__ import division   #just to have a float number from a division of integres 
# -*- coding: utf-8 -*-
"""
@author: ioanna

"""

stoplist = set('for a of the and to in a about above after again against all am an and any are aren as at be because been before being below between both but by can canno could couldn did didn do does doesn doing don down during each few for from further had hadn has hasn have haven having he he ll he s her here hers herself him himself his how i ve if in into is isn t it it s its itself let me more most mustn my myself no nor not of off on once only or other ought our our ourselves out over own same shan t she she d she ll she should shouldn so some such than that that the their theirs them themselves then there there these they they they they re they this those through to too under until up very was wasn we were weren what when wher where which while who whom why with won would wouldn you your yours yourself yourselves'.split())

import numpy
movie_plot = {}
movie_script = {}
movie_plotvocab = {}
movie_genres = {}
movie_score = {}

data_compl = open ("outputfinalscoregenre.txt", "r")
for line in data_compl:
    values = line.split (" ++$++ ") 
    movie_plot[values[0]] =values[3]    # dictionary title-plot
    movie_score[values[0]]= values[1]   # dictionary title-score
    movie_genres[values[0]]=values[2]   # dictionary title-genres
    movie_script[values[0]]=values[4]   # dictionary title-script
data_compl.close()

# define dictionaries within every genre
# comments after each line tell the number of movies belonging to this genre
movie_thriller = {} #243
movie_drama = {} #290   thriller & drama = 105
movie_action = {} #139  thriller & action = 85
movie_scifi = {} #101
movie_crime = {} #134   thriller & crime = 97
movie_horror = {} #84
movie_mystery = {} #96
movie_romance = {} #123   drama & romance = 79
movie_adventure = {} #91
movie_comedy = {} #149    comedy & romance = 60
movie_war = {} #20
movie_animation = {} #13
movie_western = {} #11
movie_documentary = {} #1
movie_sport = {} #8
movie_fantasy = {} #63

# define scores and summary dictionary
score_drama = {}
plot_drama = {}

# extract script, score and summary for "drama" movies        
for id in movie_genres.keys():
    if "drama" in movie_genres[id]:
        movie_drama[id]=movie_script[id]  #dictionary drama movie title + scrip
        score_drama[id]=float(movie_score[id])  #dictionary drama movie title + score
        plot_drama[id]=movie_plot[id] # dictionary drama movie title-plot
#build up the training set (200 movies)
score_drama_train={}
x=0
for id in score_drama.keys():
    if x<200:
        score_drama_train[id]=score_drama[id]
        x += 1
    
    
# binarise success variable for "drama" movies
success_drama_dic={}
for id in score_drama_train.keys():
    if score_drama_train[id]>7.4:
        success_drama_dic[id]=1
    else:
        success_drama_dic[id]=0


# work with the scripts
# build 2 dictionaries, one for t=0 and one for t=1
script_drama_0={} #for the not successful
script_drama_1={} #for the successful
for id in success_drama_dic.keys():
    if success_drama_dic[id] == 0:
        script_drama_0[id]=movie_drama[id]
    else:
        script_drama_1[id]=movie_drama[id]
        
from gensim import corpora 
 
#NB from now on almost everything is doubled, adding 0 or 1 to the name, corresponding
 # respectively to not successful case or successful case
    
#corpus = movie_drama.values()
corpus_0 = script_drama_0.values()
corpus_1 = script_drama_1.values()


# remove words in the stoplist

#texts = [[word for word in document.lower().split() if word not in stoplist] for document in corpus]
texts0 = [[word for word in document.lower().split() if word not in stoplist] for document in corpus_0] #for the not successful
texts1 = [[word for word in document.lower().split() if word not in stoplist] for document in corpus_1] #for the successful
# texts is a list of 290 elements, the words (filtered) for each script

# remove words which occur only once in the complete corpus

#all_tokens = sum(texts, [])
all_tokens0 = sum(texts0, [])
all_tokens1 = sum(texts1, [])

#tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
tokens_once0 = set(word for word in set(all_tokens0) if all_tokens0.count(word) == 1)
tokens_once1 = set(word for word in set(all_tokens1) if all_tokens1.count(word) == 1)

#texts = [[word for word in text if word not in tokens_once] for text in texts]
texts0 = [[word for word in text if word not in tokens_once0] for text in texts0]
texts1 = [[word for word in text if word not in tokens_once1] for text in texts1]

#dictionary = corpora.Dictionary(texts) #dictionary (i.e. the vocabulary): key=word, value=id of the wprd (number)
dictionary0 = corpora.Dictionary(texts0)
dictionary1 = corpora.Dictionary(texts1)

#rev_dic = dict((v,k) for k,v in dictionary.iteritems())  #reverse key and valuse of the dictionary
rev_dic0 = dict((v,k) for k,v in dictionary0.iteritems()) 
rev_dic1 = dict((v,k) for k,v in dictionary1.iteritems()) 

#D=len(texts)#n^o of documents
D0=len(texts0)
D1=len(texts1)
#print ("number of documents:", D)
print ("number of successful documents:", D1)
print ("number of not successful documents:", D0)
#V=len(dictionary)#length of vocabulary, cleaned: 22496 words
V0=len(dictionary0)
V1=len(dictionary1)
#print ("length of the vocabulary:", V)
print ("length of the vocabulary for successful:", V1)
print ("length of the vocabulary for not successful:", V0)

K=10 #n^o of topics

from random import randint
#z_m_n=[[ randint(0,K-1) for _ in xrange(len(d))] for d in texts] #contaiins 290 vectors (=n^o of documents), of length=length of that document
#random topic assignement for all words in all documents (initialization)
z_m_n0=[[ randint(0,K-1) for _ in xrange(len(d))] for d in texts0]
z_m_n1=[[ randint(0,K-1) for _ in xrange(len(d))] for d in texts1]
#this z_m_n matrix keeps trace of each topic assignement for each word in each document, with repetitions


#n_m_z= numpy.zeros((D,K)) #will count the n^o of words in each document assigned to a certain topic
n_m_z0=numpy.zeros((D0,K))
n_m_z1=numpy.zeros((D1,K))

n_z=numpy.zeros((K))# will count the total number of times each word assigned to each topic, globally, with repetitions
n_z0=numpy.zeros((K))
n_z1=numpy.zeros((K))

#n_z_t=numpy.zeros((K,V))#will count n^o times each word in vocabulary is assigned to topic z
n_z_t0=numpy.zeros((K,V0))
n_z_t1=numpy.zeros((K,V1))

#n_m=numpy.zeros((D)) #will count the number of words in each document
n_m0=numpy.zeros((D0))
n_m1=numpy.zeros((D1))

beta=.1  #values for Gibbs
alpha=.1



# NOT SUCCESSFUL

i=0
for m, doc in enumerate(texts0): #m: doc id (for each document)
  for n, t in enumerate(doc): #n: id of word inside document, t:  word
    # initialize the count matrices
    t=rev_dic0[t] #now t is the id of the word according to the vocabulary
    z = z_m_n0[m][n]  #current topic assignement
    n_m_z0[m][z] += 1 #here we initialize the other matrices accordingly
    n_z_t0[z,t] += 1 
    n_z0[z] += 1 
    n_m0[m] += 1 
for u in range (400): #n^o of iterations, to be increased
    for m, doc in enumerate(texts0): #m: doc id (for each document)
      for n, t in enumerate(doc): #n: id of word inside document, t: word, over all words over all documents
    # decrease counts for word t with topic z
        t=rev_dic0[t] #now t  is the id of the word according to the vocabulary
        z = z_m_n0[m][n] #current topic assignment for each word in the corpus
        n_m_z0[m][z] -= 1 #decrement the counts associated with the current assignment, on the document
        n_z_t0[z,t] -= 1 #decrement on the word assignment
        n_z0[z] -= 1 #decrement count for topic
        n_m0[m] -= 1 #decrement count for document

    # sample new topic for multinomial, taking the argmax over the probabilities of the topics               
        p_z_left0 = (n_z_t0[:, t] + beta) / (n_z0 + V0 * beta) #n_z_t[:, t]= n^o times word is assigned to topic z,for each z(array) n_z=total number of words assigned to topic z
        p_z_right0 = (n_m_z0[m] + alpha) / ( n_m0[m] + alpha * K) #n_m_z[m]= n^o words in document m assigned to topic z, n_m[m]=number of words in document m
    #the probabilties above are vectorized w.r.t. the topics 1,..,K        
        p_z0 = p_z_left0 * p_z_right0 #  phi * theta
      # print p_z_left
      # print p_z_right
        print p_z0
        p_z0 /= numpy.sum(p_z0)
        new_z = numpy.random.multinomial(1, p_z0).argmax() 

    # set z as the new topic, increment counts
        z_m_n0[m][n] = new_z
        n_m_z0[m][new_z] += 1
        n_z_t0[new_z, t] += 1
        n_z0[new_z] += 1
        n_m0[m] += 1
        i+=1
        print i
        print u
        #print "you're doin the first"

#SUCCESSFUL

r=0
for m, doc in enumerate(texts1): #m: doc id (for each document)
  for n, t in enumerate(doc): #n: id of word inside document, t:  word
    # initialize the count matrices
    t=rev_dic1[t] #now t is the id of the word according to the vocabulary
    z = z_m_n1[m][n]  #current topic assignement
    n_m_z1[m][z] += 1 #here we initialize the other matrices accordingly
    n_z_t1[z,t] += 1 
    n_z1[z] += 1 
    n_m1[m] += 1 
for u in range (400): #n^o of iterations, to be increased
    for m, doc in enumerate(texts1): #m: doc id (for each document)
      for n, t in enumerate(doc): #n: id of word inside document, t: word, over all words over all documents
    # decrease counts for word t with topic z
        t=rev_dic1[t] #now t  is the id of the word according to the vocabulary
        z = z_m_n1[m][n] #current topic assignment for each word in the corpus
        n_m_z1[m][z] -= 1 #decrement the counts associated with the current assignment, on the document
        n_z_t1[z,t] -= 1 #decrement on the word assignment
        n_z1[z] -= 1 #decrement count for topic
        n_m1[m] -= 1 #decrement count for document

    # sample new topic for multinomial, taking the argmax over the probabilities of the topics               
        p_z_left1 = (n_z_t1[:, t] + beta) / (n_z1 + V1 * beta) #n_z_t[:, t]= n^o times word is assigned to topic z,for each z(array) n_z=total number of words assigned to topic z
        p_z_right1 = (n_m_z1[m] + alpha) / ( n_m1[m] + alpha * K) #n_m_z[m]= n^o words in document m assigned to topic z, n_m[m]=number of words in document m
    #the probabilties above are vectorized w.r.t. the topics 1,..,K        
        p_z1 = p_z_left1 * p_z_right1 #  phi * theta
      # print p_z_left
      # print p_z_right
        print p_z1
        p_z1 /= numpy.sum(p_z1)
        new_z = numpy.random.multinomial(1, p_z1).argmax() 

    # set z as the new topic, increment counts
        z_m_n1[m][n] = new_z
        n_m_z1[m][new_z] += 1
        n_z_t1[new_z, t] += 1
        n_z1[new_z] += 1
        n_m1[m] += 1
        r+=1
        print r
        print u
        #print "you're doin the sec"

#PREDICT SCORE ON THE TEST SET (90 DOCUMENTS)

#we compute the likelihood for the test document according to the formula in the paper:
#  prod_{words in document d} ( sum_{topics} (words assigned to topic k in the new document/total number of words in the document) * n_z_t0[k,w]/n_z0[k]=theta * phi, for the case 0.  )
#let's start with the 201th movie

list_drama=list(movie_drama.keys())#just to acces the movie by its number

script_201=[] #all the words in document 201, with repetitions
for word in  movie_drama[list_drama[201]].lower().split():
    if word in dictionary0.values():
       script_201.append(word)
       
#texts201 = [[word for word in  movie_drama[list_drama[201]].lower().split()  if word in dictionary0.values()] ]
#dictionary_201 = corpora.Dictionary(texts201) #dictionary of the new document

theta_201=numpy.zeros((1,K))
phi_201=numpy.zeros((K, len(script_201)))

#rev_dic201 = dict((v,k) for k,v in dictionary_201.iteritems()) 

#build up phi
for w, word in enumerate(script_201):
    phi_201[:,w]=n_z_t0[:,rev_dic0[word]]/n_z0 #number of times each word in the vocabulary of the new document is assigned to a certain topic, according to the previous gibbs estimate

#now we build up theta as follows:
# -look which topic has been assigned for each word in the new document, taking the argmax of the value of phi
# -compute the proportion, by dividing by the number of words in the new document
topic_assign=[0]*len(script_201)
for i in range(len(script_201)):
        topic_assign[i]=phi_201[:,i].argmax()
   
for k in range(K):
    theta_201[0,k] = topic_assign.count(k)/len(script_201)
import math
log_lik=0
for w in range(len(script_201)):
    log_lik += math.log(numpy.dot(theta_201, phi_201[:,w]))
    
#now same with score=1
    
script_201succ=[] #all the words in document 201, with repetitions
for word in  movie_drama[list_drama[201]].lower().split():
    if word in dictionary1.values(): 
       script_201succ.append(word)
       

theta_201succ=numpy.zeros((1,K))
phi_201succ=numpy.zeros((K, len(script_201succ)))
for w, word in enumerate(script_201succ):
    phi_201succ[:,w]=n_z_t1[:,rev_dic1[word]]/n_z1 #number of times each word in the vocabulary of the new document is assigned to a certain topic, according to the previous gibbs estimate


topic_assign_succ=[0]*len(script_201succ)
for i in range(len(script_201succ)):
        topic_assign_succ[i]=phi_201succ[:,i].argmax()
   
for k in range(K):
    theta_201succ[0,k] = topic_assign_succ.count(k)/len(script_201)

log_lik_succ=0
for w in range(len(script_201succ)):
    log_lik_succ += math.log(numpy.dot(theta_201succ, phi_201succ[:,w]))        
#do the same for the remaining test documents    
#then we compare log_lik with log_lik_succ        
    
        