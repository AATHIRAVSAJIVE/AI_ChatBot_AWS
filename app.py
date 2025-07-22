from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import load_model

# Download resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Load saved files
model = load_model('chatbot_model.keras')
intents = json.load(open('intents.json', encoding='utf-8-sig'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Global variable to store user name
user_name = None

# Utility functions
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    global user_name
    if not intents_list:
        return "Sorry, I didn't understand that."

    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response.replace("{n}", user_name) if "{n}" in response and user_name else response
    return "Sorry, I couldn't find a suitable reply."

# Routes
@app.route("/")
def home():
    return render_template("index.html", greeting="Welcome to E-Commerce Support! May I know your name?")

@app.route("/get", methods=["POST"])
def chatbot_reply():
    global user_name
    msg = request.form["msg"].strip()

    # First message: Treat as name if name not yet set
    if user_name is None:
        user_name = msg.capitalize()
        return f"Nice to meet you, {user_name}! How can I help you today?"

    # Otherwise, proceed with normal prediction
    ints = predict_class(msg)
    response = get_response(ints, intents)
    return response

if __name__ == "__main__":
    app.run(debug=True)
