
# coding: utf-8
### Neural Language model implementation [Bengio, 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) in Keras
# In[2]:

import scipy.io as sio
import numpy as np
from keras.utils import to_categorical 


# In[3]:

# Load file in matlab format
mat_contents = sio.loadmat('/users/kalpeshpatel/Downloads/data.mat')


# In[3]:

# Load Training data
xx = mat_contents['data']
training = xx['trainData'][0,0]
training_x = training[0:3,].T
training_x = training_x -1
print("training_x:" + str(training_x.shape))
training_y = training[3,:].T
training_y = training_y -1
training_y_one = to_categorical(training_y)
training_y_one.shape


# In[4]:

# Load Test data

test = xx['testData'][0,0]
test_x = test[0:3,].T
test_x = test_x - 1
print("test_x:" + str(test_x.shape))
test_y = test[3].T
test_y = test_y - 1
test_y_one = to_categorical(test_y)
test_y_one.shape
test_y.shape


# In[5]:

# Load validation data

valid = xx['validData'][0,0]
valid_x = valid[0:3,]
valid_x.shape
valid_y = valid[3,]
valid_y.shape


# In[6]:

yy = xx['vocab'][0,0]
index_to_word = {}
word_to_index = {}
vocab_size = yy.shape[1]
#print("vocab size:" + str(vocab_size))
for i in range(yy.shape[1]):
    word = yy[0][i][0]
    #print(word)
    index_to_word[i] = word
    word_to_index[word] = i
#print(word_to_index['just'])
#print(index_to_word[11])


# In[7]:

cache = (training_x,training_y,yy)


# In[ ]:

# Add training and validation split from Keras


# In[8]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.optimizers import Adam


# In[9]:

# Notes for embedding: the model will take as input matrix of size (batch, input_length = 3).
# the largest integer (i.e. word index) in the input should be no larger than 250 (vocabulary size).

def buildModel(cache):
    
    training_x,training_y,vocab = cache
    vocabSize = vocab.shape[1]
    input_length = training_x.shape[1]
    vectorSize = 50
    denseLayer = 200
    model = Sequential()
    model.add(Embedding(vocabSize, vectorSize, input_length=input_length))
    model.add(Flatten())
    model.add(Dense(denseLayer,activation = 'tanh'))
    model.add(Dense(vocabSize,activation = 'softmax'))
    model.summary()
    return model


# In[11]:

from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["#", "Word1", "word2", "word3", "output", "actual"]
classes = model.predict(test_x,batch_size = 64)
for  i in range(classes.shape[0]):
    output = np.random.choice(a= yy[0],p = classes[i,:])
    table.add_row([i,index_to_word[test_x[i,0]],index_to_word[test_x[i,1]], index_to_word[test_x[i,2]],index_to_word[test_y[i]],output])
    #print(str(i) + " " + index_to_word[test_x[i,0]] + " " + index_to_word[test_x[i,1]] + " " + index_to_word[test_x[i,2]] + " " + index_to_word[test_y[i]] + " " + str(output))
#print(table) 


# In[ ]:



