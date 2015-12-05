# -*- coding: utf-8 -*-
"""
LDA Analysys with gibbs on the scripts for the DRAMA movies. For other genres, just build up 
the dictionary with moovies of that genre is a similar fashion and replace the names in
the model. After there are the regression studies.
@author: ioanna
"""
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer() #we need these elements to build the document-term matrix and objects linked to it
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english")

import numpy as np
import lda

movie_plot = {}
movie_script = {}
movie_plotvocab = {}
movie_genres = {}
movie_score = {}

data_compl = open ("outputfinalscoregenre.txt", "r") #dataset

#building up the dictionaries

for line in data_compl:
    values = line.split (" ++$++ ") 
    movie_plot[values[0]] =values[3]  #dictionary title-plot
    movie_score[values[0]]= values[1]
    movie_genres[values[0]]=values[2]
    movie_script[values[0]]=values[4]   #dicionary title-script
data_compl.close()

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

#define scores dictionary

score_drama ={}

for id in movie_genres.keys():
    if "drama" in movie_genres[id]:
        movie_drama[id]=movie_script[id]  #dictionary drama movie title + script
        score_drama[id]=movie_score[id]  #dictionary drama movie title + score

corpus = movie_drama.values()

X = vectorizer.fit_transform(corpus) #document-term matrix
print("type(X): {}".format(type(X)))
print("shape: {}\n".format(X.shape))

vocab = np.array(vectorizer.get_feature_names()) #vocabulary
print("type(vocab): {}".format(type(vocab)))
print("len(vocab): {}\n".format(len(vocab)))

titles = movie_drama.keys() #titles
print("type(titles): {}".format(type(titles)))
print("len(titles): {}\n".format(len(titles)))

# fit the LDA model with Gibbs.
# IMPORTANT NOTE: here we use the LDA package instead of our own derivation, just for performance reasons

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)  
topic_word = model.topic_word_ 

#top words for each document
n_top_words = 20
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Language {}: {}'.format(i, ' '.join(topic_words)))
    
#document-topic distributions
doc_topic = model.doc_topic_
for i in range(290):
    print("{} (top language: {})".format(titles[i], doc_topic[i].argmax()))

#some plots
import matplotlib.pyplot as plt

try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass   

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0,2,4,6,8]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 21)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Movie {}".format(k))

ax[4].set_xlabel("Latent z")

plt.tight_layout()
plt.show()
print("type(doc_topic): {}".format(type(doc_topic)))
print("len(doc_topic): {}\n".format(len(doc_topic)))

#REGRESSION STUDIES

#floating the scores
scores = [float(x) for x in score_drama.values()]
scores= np.asarray(scores)

#linear regression
#train: 200 movies, test: 90 movies

#with statsmodels

import statsmodels.api as sm
import matplotlib.pyplot as plt
res = sm.OLS(scores[0:200], doc_topic[0:200]).fit()  
print res.summary()
print res.params

fig = plt.figure()
fig = sm.graphics.plot_partregress_grid(res, fig=fig)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(res, "x1", fig=fig)

#with sklearn

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
model_linreg = LinearRegression(fit_intercept=False)
model_linreg= model_linreg.fit(doc_topic[0:200], scores[0:200] )
 
print model_linreg.coef_
print model_linreg.predict(doc_topic[201:290])
print scores[201:290]

# Plot outputs

scores_test=scores[201:290]
scores_testarr=np.array(scores_test)
pred_test=model_linreg.predict(doc_topic[201:290])
pred_ord=[x for (y,x) in sorted(zip(scores_test,pred_test))]

import matplotlib.pyplot as plt

plt.plot(np.arange(89),  sorted(scores_testarr),'ro',label='true scores')
plt.plot(np.arange(89), pred_ord, 'b--',label='predicted scores')
plt.legend()
plt.ylabel('scores')
plt.xlabel('test movies')
plt.show()

#LOGISTIC REGRESSION

#now we use bynary target variable (success/not success) 
success_drama = [0]*290
for i in range(290):
    if scores[i] > 7.4:
        success_drama[i] = 1
success_drama = np.asarray(success_drama) # 0:not successful, 1:successful

from sklearn.linear_model import LogisticRegression

model_log = LogisticRegression()
model_log = model_log.fit(doc_topic[0:200], success_drama[0:200] )
print model_log.score(doc_topic[0:200], success_drama[0:200]) #accuracy on the total dataset

# 10-fold cross validation

accuracy_scores = cross_val_score(LogisticRegression(fit_intercept=False), doc_topic, success_drama,scoring='accuracy',  cv=10)

#plots

model_log.predict(doc_topic[0:290])
scores_test=scores[201:290]
scores_testarr=np.array(scores_test)
pred_testsucc=model_log.predict(doc_topic[201:290])
pred_ordsucc=[x for (y,x) in sorted(zip(scores_test,pred_testsucc))]

import matplotlib.pyplot as plt

plt.plot(np.arange(89), sorted(scores_testarr),'ro',label='true success')
plt.plot(np.arange(89), pred_ordsucc, 'b--',label='predicted success')
plt.legend()
plt.ylabel('success')
plt.xlabel('test movies')
plt.show()

# SVM with sklearn

from sklearn import svm

clf = svm.SVC()
mode_svm=clf.fit(doc_topic[0:200], success_drama[0:200]) 
accuracy_scoressvm = cross_val_score(svm.SVC(), doc_topic, success_drama,scoring='accuracy',  cv=10)
print accuracy_scoressvm
clf.predict(doc_topic)


#new parameters
#clf2 = svm.SVC(C=100, gamma=0.001)
#mode_svm2=clf2.fit(doc_topic[0:290], success_drama[0:290]) 
#accuracy_scoressvm2 = cross_val_score(svm.SVC(C=100, gamma=0.001), doc_topic, success_drama,scoring='accuracy',  cv=10)
#print accuracy_scoressvm2
#clf2.predict(doc_topic[0:290])

# SVR with sklearn

clfsvr = svm.SVR()
clfsvr.fit(doc_topic[0:200], scores[0:200]) 
clfsvr.predict(doc_topic[0:290])

# non-linear regression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
doc_topic_poly=poly.fit_transform(doc_topic)
clfridge = Ridge()
clfridge.fit(doc_topic_poly, scores) 
clfridge.predict(doc_topic_poly)


