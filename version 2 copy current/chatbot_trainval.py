import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
import json
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import numpy as np
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split  # Import train_test_split

with open("intents2.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Tokenizing each pattern in our dataset
# Adding to list of tokenize patterns or words and their labels or tag
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Reducing each word to their root word and sorting them
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# Inserting list of words and labels into pickle file 
# Performing serialization
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# Encoding our input to bag of words
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Creating bag of words encoding of dataset
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(training, output, test_size=0.2, random_state=42)

# Building deep neural networks
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))



# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the model with validation data
hist = model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),  # Using validation data during training
                 epochs=1000, 
                 batch_size=8, 
                 verbose=1)
