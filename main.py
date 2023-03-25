import random
import pickle

import numpy as np
import pandas as pd
from keras.optimizers import RMSprop
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM

text_df = pd.read_csv ('fake_or_real_news.csv')

text = list (text_df.text.values)
joined_text = ' '.join (text)

partial_text = joined_text [:10000]

tokenizer = RegexpTokenizer (r"\w+")
tokens = tokenizer.tokenize (partial_text.lower ())

# Create sequences of tokens
unique_tokens = np.unique (tokens)
unique_token_index = {token: idx for idx, token in enumerate (unique_tokens)}

n_words = 10
input_words = []
next_words = []

for i in range (len (tokens) - n_words):
	input_words.append (tokens [i:i + n_words])
	next_words.append (tokens [i + n_words])

# Create X and y
X = np.zeros ((len (input_words), n_words, len (unique_tokens)), dtype = bool)
y = np.zeros ((len (input_words), len (unique_tokens)), dtype = bool)

for i, input_word in enumerate (input_words):
	for j, word in enumerate (input_word):
		X [i, j, unique_token_index [word]] = 1
	y [i, unique_token_index [next_words [i]]] = 1

model = Sequential ()
model.add (LSTM (128, input_shape = (n_words, len (unique_tokens)), return_sequences = True))
model.add (LSTM (128))

model.add (Dense (len (unique_tokens)))
model.add (Activation ('softmax'))

model.compile (loss = 'categorical_crossentropy', optimizer = RMSprop (learning_rate = 0.01), metrics = ['accuracy'])
model.fit (X, y, batch_size = 128, epochs = 30, shuffle = True)

model.save ('my_model.h5')

model = load_model ('my_model.h5')


def predict_next_word (input_text, n_best):
	input_text = input_text.lower ()
	X = np.zeros ((1, n_words, len (unique_tokens)))
	for i, word in enumerate (input_text.split ()):
		X [0, i, unique_token_index [word]] = 1

	predictions = model.predict (X) [0]
	return np.argpartition (predictions, -n_best) [-n_best:]


# possible_next_words = predict_next_word ('the president', 10)

def generate_text (start_text, text_length, creativity = 3):
	word_list = start_text.split ()
	current = 0
	for _ in range (text_length):
		sub_sequence = " ".join (tokenizer.tokenize (" ".join (word_list).lower ()) [current:current + n_words])
		try:
			choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
		except:
			choice = random.choice (unique_tokens)
		word_list.append (choice)
		current += 1
	return " ".join (word_list)


generate_text ('the president', 100, 3)
