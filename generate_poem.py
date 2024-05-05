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
