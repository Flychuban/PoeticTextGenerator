import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the data
with open('./data/shakespeare.txt', 'rb') as f:
    text_binary = f.read().decode(encoding='utf-8').lower()

# Create a dictionary of characters
characters_dictionary = sorted(set(text_binary))
char_to_index = dict((c, i) for i, c in enumerate(characters_dictionary))
index_to_char = dict((i, c) for i, c in enumerate(characters_dictionary))

# Predict the next character
SEQUENCE_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

for i in range(0, len(text_binary) - SEQUENCE_LENGTH, STEP_SIZE):
    sentences.append(text_binary[i: i + SEQUENCE_LENGTH])
    next_characters.append(text_binary[i + SEQUENCE_LENGTH])

# Create the input and output tensors
x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters_dictionary)), dtype=bool)
y = np.zeros((len(sentences), len(characters_dictionary)), dtype=bool)

# Encode the characters which are present in the text into the tensors x and y
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1
    
# Building the model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters_dictionary)), activation='tanh'))
model.add(Dense(len(characters_dictionary), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=10)

# Save the model
model.save('./models/shakespeare_model.h5')