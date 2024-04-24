import pickle
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from keras.models import load_model
import json
import random

app = Flask(__name__)

# Load the trained model
model = load_model('chatbot_model.h5')

# Load intents JSON file
intents_json = json.loads(open('intents2.json').read())

# Load words and labels
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

# Initialize Lancaster Stemmer
stemmer = LancasterStemmer()

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

# Function to predict class/intent
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response
def get_response(intents, intents_json):
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = {
                "response": random.choice(i['responses']),
                "precautions": i.get('precautions', []),
                "treatments": i.get('treatments', []),
            }
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']
    intents = predict_class(user_message, model)
    response_info = get_response(intents, intents_json)
    return jsonify(response_info)

if __name__ == '__main__':
    app.run(port=3000)
