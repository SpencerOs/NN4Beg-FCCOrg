from keras.datasets import imdb
from keras.utils import pad_sequences
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAX_LEN = 800  # actual max of 2494
BATCH_SIZE = 128

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
train_data = pad_sequences(train_data, MAX_LEN)
test_data = pad_sequences(train_data, MAX_LEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=3, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print("Results: {}".format(results))

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.psd_sequences([tokens], MAX_LEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

# make a decode (int -> word) function
reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]
print(decode_integers(encoded))

# now time to make a prediction
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text(text)
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was great! I really loved it and can not wait to watch it again."
predict(positive_review)

positive_review = "That movie was not too great. I don't think I could stomach another viewing of it."
predict(negative_review)