from flask import Flask, request, jsonify
from flask_cors import CORS  # Used to handle cross-origin requests

import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import random

# Load the pre-trained word embeddings from the glove file
embeddings_index = {}
with open('glove.42B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# Load predefined bot intents and their patterns/responses
with open('intentsFile.json', 'r') as f:
    data = json.load(f)

texts = []
labels = []

# Populate texts and labels from the intents data
for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern.lower())
        labels.append(intent['tag'])

# Tokenize and sequence the patterns
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

# Label encode the intent tags
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Create an embedding matrix for the tokenized patterns
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((len(texts), maxlen, embedding_dim))
for i, sequence in enumerate(padded_sequences):
    for j, word_index in enumerate(sequence):
        if word_index != 0:
            embedding_vector = embeddings_index.get(tokenizer.index_word[word_index])
            if embedding_vector is not None:
                embedding_matrix[i][j] = embedding_vector

# Load the pre-trained intent classification model
loaded_model = load_model('intentClassificationBestModel.h5')

def classify_intent(user_input):
    """Classify user input into one of the predefined intents."""
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded = pad_sequences(user_sequence, maxlen=maxlen, padding='post')

    user_embedding = np.zeros((1, maxlen, embedding_dim))
    for i, word_index in enumerate(user_padded[0]):
        if word_index != 0:
            embedding_vector = embeddings_index.get(tokenizer.index_word[word_index])
            if embedding_vector is not None:
                user_embedding[0][i] = embedding_vector

    prediction = loaded_model.predict(user_embedding)
    predicted_label = np.argmax(prediction, axis=1)[0]
    intent = encoder.inverse_transform([predicted_label])[0]

    return intent

def get_response(predicted_intent):
    """Retrieve a response for a predicted intent."""
    for intent in data['intents']:
        if intent['tag'] == predicted_intent:
            return intent['responses']

        
app = Flask(__name__)
CORS(app)  # Handle CORS issues for front-end to back-end communication

@app.route('/predict', methods=['POST'])
def predict():
    """Handle incoming user messages and respond based on intent."""
    user_input = request.json['message']
    
    # Classify and respond to other user inputs
    predicted_intent = classify_intent(user_input)
    response = get_response(predicted_intent)
    return jsonify({"response": response[0]})

if __name__ == '__main__':
    app.run(debug=True)
