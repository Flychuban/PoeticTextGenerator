import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop

# Load the data
with open('./data/shakespeare.txt', 'rb') as f:
    text_binary = f.read().decode(encoding='utf-8').lower()

# Create a dictionary of characters
characters_dictionary = sorted(list(set(text_binary)))
char_to_index = dict((c, i) for i, c in enumerate(characters_dictionary))
index_to_char = dict((i, c) for i, c in enumerate(characters_dictionary))

# Predict the next character
SEQUENCE_LENGTH = 50
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