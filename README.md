# AI_ChatBot_AWS
## E-Commerce AI Chatbot Hosted on AWS

This project implements an AI-powered e-commerce support chatbot using TensorFlow and Flask and hosted on AWS EC2, designed to understand customer queries and provide intelligent responses. The chatbot is trained on a custom intents.json dataset, which includes expanded pattern coverage and a wide variety of real-world e-commerce intents—such as order tracking, shipping info, return policies, payment methods, complaints, and more.

The backend is built using Natural Language Processing (NLP) techniques:

Tokenization, lemmatization, and Bag-of-Words encoding

A neural network model built with TensorFlow and Keras to classify user input into defined intent categories

Model training generates chatbot_model.keras, words.pkl, and classes.pkl for inference

A Flask web server integrates the trained model with a lightweight web interface:

The frontend is built with HTML and CSS, styled for a smooth chat experience

All user queries are processed via Flask routes, and the bot responds in real time

For production deployment, the full application is hosted on AWS EC2, enabling:

Secure and publicly accessible interface

Scalable infrastructure for testing or expansion

Easy upload via GitHub + EC2 deployment


