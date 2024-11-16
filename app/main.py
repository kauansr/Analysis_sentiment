import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import re
from spacy.lang.en.stop_words import STOP_WORDS


modelo = tf.keras.models.load_model('my_model.keras')


tokenizer = Tokenizer()

app = FastAPI()

stop_words = STOP_WORDS

classes = ['negativo', 'neutro', 'positivo']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def interpretar_predicao(predicao, limiar=0.5):
    indice_maximo = np.argmax(predicao)
   
    if predicao[indice_maximo] >= limiar:
        return classes[indice_maximo]
    else:
        return 'neutro'

class TextoEntrada(BaseModel):
    texto: List[str]


@app.post("/predict")
async def predicao(data: TextoEntrada):
    textos_processados = [preprocess_text(texto) for texto in data.texto]
    
    tokenizer.fit_on_texts(textos_processados)
    sequences = tokenizer.texts_to_sequences(textos_processados)
    X = pad_sequences(sequences, padding='post')

    predicoes = modelo.predict(X)

    print(predicoes)

    sentimentos = [interpretar_predicao(pred, limiar=0.4) for pred in predicoes]

    return {"predicoes": sentimentos}