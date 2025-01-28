import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import re
from spacy.lang.en.stop_words import STOP_WORDS
import pickle


# Load the trained model
modelo = tf.keras.models.load_model('model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# List of stop words in English
stop_words = STOP_WORDS

# Sentiment classes for prediction: 'negative', 'neutral', 'positive'
classes = ['negative', 'neutral', 'positive']

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Converting to lowercase
    - Removing URLs
    - Removing mentions (e.g., @username)
    - Removing hashtags
    - Removing punctuation
    - Removing emojis
    - Removing stop words

    Args:
    - text (str): The input text to preprocess.

    Returns:
    - str: The cleaned and preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'[^\x00-\x7F]+', '', text) # This regex removes any character that is not ASCII, which includes emojis and other non-ASCII characters.
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

def interpret_prediction(prediction, threshold=0.5):
    """
    Interprets the model's prediction based on a threshold.

    Args:
    - prediction (list): The prediction array from the model.
    - threshold (float): The minimum confidence level to classify the sentiment.

    Returns:
    - str: The sentiment label ('negative', 'neutral', 'positive', or 'neutral' if below threshold).
    """
    max_index = np.argmax(prediction)
   
    if prediction[max_index] >= threshold:
        return classes[max_index]
    else:
        return 'neutral'  # Default to 'neutral' if confidence is below the threshold

class TextInput(BaseModel):
    """
    Pydantic model to parse input data.

    Attributes:
    - texto (List[str]): List of texts to analyze.
    """
    texto: List[str]

@app.post("/predict")
async def prediction(data: TextInput):
    """
    Predicts the sentiment of input texts by:
    - Preprocessing the texts
    - Tokenizing the texts
    - Predicting sentiment using the trained model

    Args:
    - data (TextInput): The input data containing a list of texts.

    Returns:
    - dict: A dictionary containing the predicted sentiments for each text.
    """
    # Preprocess the input texts
    processed_texts = [preprocess_text(text) for text in data.texto]
    
    # Tokenize the preprocessed texts
    sequences = tokenizer.texts_to_sequences(processed_texts)
    X = pad_sequences(sequences, padding='post', maxlen=100)

    # Make predictions using the model
    predictions = modelo.predict(X)

    # Interpret the predictions and classify sentiment
    sentiments = [interpret_prediction(pred, threshold=0.5) for pred in predictions]

    return {"predictions": sentiments}