from keras.models import Sequential
from keras.layers import Dense, LSTM , Embedding
from keras.preprocessing import sequence
from pythainlp import word_tokenize
import collections
import pandas as pd
import numpy as np
import os

directory = os.path.dirname(os.path.realpath(__file__))

def load_train():
    
    datasets_dir = directory + '/datasets/train/'
    datasets = np.empty((1,2) , dtype = 'str')
    
    for file in os.listdir(datasets_dir):
        print (file)
        file_dir = os.path.join(datasets_dir, str(file))
        file_dir = file_dir.replace('\\' , '/')
        data = pd.read_csv(str(file_dir)).iloc[:,0:2].values.astype('str')
        datasets = np.concatenate((datasets, data), axis=0)
        
    datasets = datasets[1: , 0:]
    
    X = datasets[: , 1]
    y = datasets[: , 0]
    
    return X,y.reshape(-1,1)

def load_test():
    
    datasets_dir = directory + '/datasets/test/'
    datasets = np.empty((1,2) , dtype = 'str')
    
    for file in os.listdir(datasets_dir):
        print (file)
        file_dir = os.path.join(datasets_dir, str(file))
        file_dir = file_dir.replace('\\' , '/')
        data = pd.read_csv(str(file_dir)).iloc[:,0:2].values.astype('str')
        datasets = np.concatenate((datasets, data), axis=0)
        
    datasets = datasets[1: , 0:]
    
    X = datasets[: , 1]
    y = datasets[: , 0]
    
    return X,y.reshape(-1,1)

def make_word_tokenize (datas):
    
    list_data = []
    
    for data in datas :
        word_tokenized = word_tokenize(data , engine = 'deepcut')
        list_data.append(word_tokenized)
    
    return list_data

def make_dictionary (X_train , X_test) : 
    count = collections.Counter('')
    for data in X_train :
        count += collections.Counter(data)
        
    for data in X_test :
        count += collections.Counter(data)
    
    count = count.most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return dictionary , reverse_dictionary

def int_represent_word (list_data,dictionary) :
    keys = dict.keys(dictionary)
    for data in list_data:
        for word in range(len(data)):
            if data[word] in keys:
                data[word] = dictionary[data[word]]
            else:
                data[word] = dictionary[reverse_dictionary['0']]
    return list_data

X_train , y_train = load_train()
X_test , y_test = load_test()

X_train = make_word_tokenize(X_train)
X_test = make_word_tokenize(X_test)

dictionary , reverse_dictionary = make_dictionary(X_train , X_test)

X_train = int_represent_word (X_train,dictionary)
X_test = int_represent_word (X_test,dictionary)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

top_words = len(dictionary)
embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(500))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

#y_pred = model.predict(X_test)            
#y_pred_classes = model.predict_classes(X_test)
#y_pred_prob = model.predict_proba(X_test)            
            


