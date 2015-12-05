# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07 11:01:38 2015

@author: ioanna

APPENDIX: to be glued on the adjusted_approach.py from line 259. It is the exactly same code, just 
with an additional loop over the 90 test movies and a more compact layout. Finally there is just the code
for the accuracy and for the plot.
NB it is not compiling by itself
"""
#PREDICT SCORE ON THE TEST SET (90 DOCUMENTS)

#we compute the likelihood for the test document according to the formula
#  prod_{words in document d} ( sum_{topics} (words assigned to topic k in the new document/total number of words in the document) * n_z_t0[k,w]/n_z0[k]=theta * phi. Theta is probably wrong.  )
#let's start with the 201th movie
from __future__ import division 
import math
list_drama=list(movie_drama.keys())#just to acces the movie by its number
for a in range(89):
    a = a+201
    script_201=[]
    for word in  movie_drama[list_drama[a]].lower().split():
        if word in dictionary0.values():
           script_201.append(word)
    theta_201=numpy.zeros((1,K))
    phi_201=numpy.zeros((K, len(script_201)))
    for w, word in enumerate(script_201):
        phi_201[:,w]=n_z_t0[:,rev_dic0[word]]/n_z0 
    topic_assign=[0]*len(script_201)
    for i in range(len(script_201)):
        topic_assign[i]=phi_201[:,i].argmax()
    for k in range(K):
        theta_201[0,k] = topic_assign.count(k)/len(script_201)
    log_lik=0
    for w in range(len(script_201)):
        log_lik += math.log(numpy.dot(theta_201, phi_201[:,w]))
    script_201succ=[] #all the words in document 201, with repetitions
    for word in  movie_drama[list_drama[201]].lower().split():
        if word in dictionary1.values(): 
           script_201succ.append(word)
    theta_201succ=numpy.zeros((1,K))
    phi_201succ=numpy.zeros((K, len(script_201succ)))
    for w, word in enumerate(script_201succ):
        phi_201succ[:,w]=n_z_t1[:,rev_dic1[word]]/n_z1 
    topic_assign_succ=[0]*len(script_201succ)
    for i in range(len(script_201succ)):
        topic_assign_succ[i]=phi_201succ[:,i].argmax()
    for k in range(K):
        theta_201succ[0,k] = topic_assign_succ.count(k)/len(script_201)
    log_lik_succ=0
    for w in range(len(script_201succ)):
        log_lik_succ += math.log(numpy.dot(theta_201succ, phi_201succ[:,w]))        
    print log_lik
    print log_lik_succ
    print a
#then we compare log_lik with log_lik_succ        
    
predictions = numpy.array( [0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0 ] )
# we wrote the prediciotns vector by hand starting from the values log_lik and log_lik_succ
#for each movie, obtained before. Just because we forgot to do that inside the previous loop so the values were not stored.
score_drama_test={}
x=0
for id in score_drama.keys():
    if x>200:
        score_drama_test[id]=score_drama[id]
    x += 1
    
    
# binarise success variable for "drama" movies
success_drama_test={}
for id in score_drama_test.keys():
    if score_drama_test[id]>7.4:
        success_drama_test[id]=1
    else:
        success_drama_test[id]=0
        

scores_test=score_drama_test.values()
scores_testarr=numpy.array(scores_test)
pred_testsucc=predictions

pred_ordsucc=[l for (y,l) in sorted(zip(scores_test,pred_testsucc))]
import matplotlib.pyplot as plt

plt.plot(numpy.arange(89), sorted(scores_testarr),'ro',label='true success')
plt.plot(numpy.arange(89), pred_ordsucc, 'b--',label='predicted success')


plt.legend(loc=0)
plt.ylabel('success')
plt.xlabel('test movies')
plt.show()

true_success_drama=numpy.array(success_drama_test.values())
test = 0
for i in range(89):
    if true_success_drama[i] == predictions[i]:
        test += 1
accuracy_test= test/89 #=0.57
    
