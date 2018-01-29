import pickle
import heapq
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

# To silence tf compilation warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Getting back the model and variables
with open('objs.pkl', 'rb') as f:
    chars, char_indices, indices_char, sequence_length = pickle.load(f)
model = load_model('keras_model2.h5')
history = pickle.load(open("history2.p", "rb"))


# Create a tensor with shape (1, #sequence length, #of unique chars),
# initialize with zeros.
# For each character in the text, pass 1.
def reshape_input(text):
    inp = np.zeros((1, sequence_length, len(chars)))
    for i, char in enumerate(text):
        inp[0, i, char_indices[char]] = 1.
    return inp


# returns the next n most probable autocompletion predictions
# from most likely to the least
def rank_predictions(predictions, n):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions)
    predictions = np.exp(predictions) / np.sum(np.exp(predictions))
    return heapq.nlargest(n, range(len(predictions)), predictions.take)


# next character is predicted until space is predicted
# this can be extended to a punctuation mark or a specific character
def complete_word(text):
    original_text = text
    completion_text = ''
    while True:
        inp = reshape_input(text)
        predictions = model.predict(inp, verbose=0)[0]
        # get the most probable char prediction:
        next_index = rank_predictions(predictions, n=1)[0]
        next_char = indices_char[next_index]
        # update the text plugged into the reshape_input:
        text = text[1:] + next_char
        completion_text += next_char
        if len(original_text + completion_text) + 2 > \
           len(original_text) and next_char == ' ':
            return completion_text


# predicts n different word completions
def predict_completions(text, n):
    inp = reshape_input(text)
    predictions = model.predict(inp, verbose=0)[0]
    # get the n most probable next char predictions:
    next_indices = rank_predictions(predictions, n)
    # for each n, complete the word:
    return [indices_char[idx] + complete_word(text[1:] +
            indices_char[idx]) for idx in next_indices]


test_sentences = [
    "I'm running late to the meeting. I'll talk to you later. Bye.",
    "What does not kill you makes you stronger.",
    "I am so tired. I just want to go home and sleep.",
    "I am so hungry. When is the dinner ready?",
    "I want to watch the derby game. Turn on the TV.",
    # This last one is exactly from the training data:
    "he will to overcome an emotion, is ultimately only the will of another, \
     or of several other, emotions."
]


for sentence in test_sentences:
    text = sentence[:40].lower()
    print(text)
    print(predict_completions(text, 3))
    print()

K.clear_session()