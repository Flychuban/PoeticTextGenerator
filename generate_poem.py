import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

SEQUENCE_LENGTH = 40
STEP_SIZE = 3


def read_data_text(data_filename):
    with open(data_filename, 'rb') as f:
        text_binary = f.read().decode(encoding='utf-8').lower()
    return text_binary

def create_char_dictionary(text_binary):
    # Create a dictionary of characters
    characters_dictionary = sorted(set(text_binary))
    char_to_index = dict((c, i) for i, c in enumerate(characters_dictionary))
    index_to_char = dict((i, c) for i, c in enumerate(characters_dictionary))
    return characters_dictionary, char_to_index, index_to_char


def create_model(data_filename, save_model_filename):
    # Load the data
    text_binary = read_data_text(data_filename)

    # Create a dictionary of characters
    characters_dictionary, char_to_index, index_to_char = create_char_dictionary(text_binary)

    # Predict the next character
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
    model.save(save_model_filename)
    
    return model

# Predict the next character based on the predictions and the temperature
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    # Load the model
    model = tf.keras.models.load_model('./models/shakespeare_model.h5')
    
    # Load the data
    text_binary = read_data_text('./data/shakespeare.txt')
    
    # Create a dictionary of characters
    characters_dictionary, char_to_index, index_to_char = create_char_dictionary(text_binary)
    
    # Pick a random sentence from the text
    start_index = random.randint(0, len(text_binary) - SEQUENCE_LENGTH - 1)
    
    generate_text = ''
    sentence = text_binary[start_index: start_index + SEQUENCE_LENGTH]
    
    generate_text += sentence
    
    # Generate the text
    for i in range(length):
        x_pred = np.zeros((1, SEQUENCE_LENGTH, len(characters_dictionary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        
        generate_text += next_char
        sentence = sentence[1:] + next_char
    
    return generate_text

# Test the generate_text function
print("---------------------- 0.2")
print(generate_text(400, 0.2))

print("---------------------- 0.35")
print(generate_text(400, 0.35))

print("---------------------- 0.5")
print(generate_text(600, 0.5))

print("---------------------- 0.75")
print(generate_text(400, 0.75))

print("---------------------- 1.0")
print(generate_text(400, 1.0))