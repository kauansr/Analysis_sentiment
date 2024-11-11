import pandas as pd
import numpy as np
import spacy
import en_core_web_lg
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spacy.lang.en.stop_words import STOP_WORDS

training_data = pd.read_csv('/content/twitter_training.csv', header=None)
print(training_data.head(10))

training_data.drop([0,1], axis=1, inplace=True)
training_data.columns = ['sentiment', 'text']
print(training_data.tail(5))

training_data["sentiment"] = np.where(training_data["sentiment"] == "Positive", 1, 0)

sns.countplot(training_data, x='sentiment')

print(training_data['sentiment'].unique())

print(training_data['sentiment'].value_counts())

counts = training_data['sentiment'].value_counts()

counts.plot(kind='bar', color='skyblue')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Value Counts of Sentiments')
plt.show()

stop_words = STOP_WORDS

nlp = en_core_web_lg.load()
print(nlp)

def preprocessamento(texto):
  texto = texto.lower()

   # Nome do usuário
  texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)

  # URLs
  texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)

  # Espaços em branco
  texto = re.sub(r" +", ' ', texto)



  documento = nlp(texto)

  lista = []
  for token in documento:
    lista.append(token.lemma_)


  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista

training_data['text'] = training_data['text'].astype(str)

training_data['text'] = training_data['text'].apply(preprocessamento)
print(training_data.head(10))

max_features = 10000

word_tokenizer = Tokenizer(
    oov_token="<OOV>",
    lower=True,
    num_words=max_features,
)

word_tokenizer.fit_on_texts(training_data["text"])

x_sequences = word_tokenizer.texts_to_sequences(training_data["text"])

x_padded = pad_sequences(x_sequences, padding="post", truncating="post", maxlen=100)

x_train, y_train = x_padded, training_data['sentiment']

print(x_train.shape, y_train.shape)

val_data = pd.read_csv("twitter_validation.csv", header=None)
print(val_data.head(10))

val_data.drop([0,1], axis=1, inplace=True)
val_data.columns = ['sentiment', 'text']
print(val_data.head(10))

print(val_data.tail(5))

val_data["sentiment"] = np.where(val_data["sentiment"] == "Positive", 1, 0)

val_data['text'] = val_data['text'].astype(str)
val_data["text"] = val_data["text"].apply(preprocessamento)

x_val_sequences = word_tokenizer.texts_to_sequences(val_data["text"])
x_val_padded = pad_sequences(x_val_sequences, padding="post", truncating="post", maxlen=100)
y_val = val_data["sentiment"]

max_features = len(word_tokenizer.word_index) + 11
output_dim = 64

model = Sequential([
    Embedding(input_dim=max_features, output_dim=output_dim),
    LSTM(64, return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=40, validation_data=(x_val_padded, y_val), callbacks=[early_stopping])

loss, acc = model.evaluate(x_val_padded, y_val)

print(f"Model Loss: {loss}\nModel Accuracy: {acc}")

history_dict = history.history

print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()