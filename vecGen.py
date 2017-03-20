from os import listdir
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
import string
import multiprocessing
from tqdm import tqdm, trange
import operator
import pickle

#files
trainDir = "aclimdb/train/"
testDir = "aclimdb/test/"
contenttrain = []
labeltrain = []
contenttest = []
labeltest = []

#extracting data
for i in listdir(trainDir+'pos'):
    contenttrain.append(open(trainDir+'pos/'+i,'r').read())
    labeltrain.append('1')
for i in listdir(trainDir+'neg'):
    contenttrain.append(open(trainDir+'neg/'+i,'r').read())
    labeltrain.append('0')
for i in listdir(testDir+'pos'):
    contenttest.append(open(testDir+'pos/'+i,'r').read())
    labeltest.append('1')
for i in listdir(testDir+'neg'):
    contenttest.append(open(testDir+'neg/'+i,'r').read())
    labeltest.append('0')
    
#shuffling
trainData = pd.DataFrame({"content":contenttrain,"label":labeltrain})
trainData = trainData.iloc[np.random.permutation(len(trainData))]
testData = pd.DataFrame({"content":contenttest,"label":labeltest})
testData = testData.iloc[np.random.permutation(len(testData))]

#pickle
testData.to_pickle("model/test.pkl")
trainData.to_pickle("model/train.pkl")

#creting tokens
def preprocessSentence(sent):
    sent =  sent[0]
    tokens = ''.join(('' if i.isdigit() else i) for i in (sent) if (i not in string.punctuation) ).lower()
    return nltk.word_tokenize(tokens)

train_sentences = list(np.array(trainData[['content']]))
train_sentences_all = []
test_sentences_all = []
for i in train_sentences:
    train_sentences_all.append(preprocessSentence(i))

#word to vec
cores = multiprocessing.cpu_count()
model_train = Word2Vec(train_sentences_all,size=100, window=5, min_count=1, workers=cores)
model_train.save('model/model_train.model')

#padding
max_index, max_value = max(enumerate(i for i in train_sentences_all), key=operator.itemgetter(1))
for i in train_sentences_all:
    i+=list('0'*(len(max_value)-len(i)))
    
#gengerate the veectors    
trainData = []
for i in range(len(train_sentences_all)):
    temp = []
    for j in train_sentences_all[i]:
        if(j=='0'):
            temp.append({'vecs':np.array(list(map(int,str(0)*100))),'label':labeltrain[i]})
        else:
            temp.append({'vecs':model_train[j],'label':0})
    trainData.append(temp)
    
with open('model/trainDataPickle.pkl','wb') as file:
    pickle.dump(trainData,file,pickle.HIGHEST_PROTOCOL)
print("finished");