'''
    Models the training text data with LSTM,
    predicts the following word in the chat box given the past 40 characters.
    Mission is to expedite the typing process
    as well as showing the corrected version of the misspelled words.
'''

import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

class Autocomplete:

    def __init__(self):
        # to divide the text into chunks of 40 characters,
        # shift each sequence by 3 characters.
        self.sequence_length = 40
        self.shift_sequence = 3
        with open("data.txt") as f:
            self.text = f.read().lower()
        # Read the text and map all the unique characters to an index
        # and all indices to a unique character:
        self.chars = sorted(list(set(self.text)))
        self.char_indices = dict((char, i) for i, char in enumerate(self.chars))
        self.indices_char = dict((i, char) for i, char in enumerate(self.chars))


    def batchSampling(self):
        # to divide the text into chunks of 40 characters (sentence),
        # shift each sequence by 3 characters.
        # store the next character for each and every sentence
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.sequence_length, self.shift_sequence):
            sentences.append(self.text[i: i + self.sequence_length])
            next_chars.append(self.text[i + self.sequence_length])

        # Generate the features and labels as an array of booleans
        # features has a size: #of training examples(sentences) x sentence length x #of unique chars
        # labels has a size: #of training examples(sentences) x #of unique chars
        features = np.zeros((len(sentences), self.sequence_length, len(self.chars)), dtype=np.bool)
        labels = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                features[i, j, self.char_indices[char]] = 1
            labels[i, self.char_indices[next_chars[i]]] = 1
        return features, labels


    def trainLSTM_model(self, features, labels):
        self.model = Sequential()
        # LSTM first layer with 128 neurons
        # Input shape: sequence_length(sentence length) x #of unique chars
        self.model.add(LSTM(128, input_shape=(self.sequence_length, len(self.chars))))
        # TO DO:
        # TRY DROPOUT LAYER, TEST WITH DIFFERENT OPTIMIZATION FUNCTIONS, EPOCHS, LEARN RATES AND
        # VALIDATION SPLIT!
        # Add a fully connected layer of length equal to number of unique chars:
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = self.model.fit(features, labels, validation_split=0.05, batch_size=128, epochs=1, shuffle=True).history
        # save the lstm model for saving time.
        self.model.save('keras_model2.h5')
        pickle.dump(history, open("history2.p", "wb"))
        # save the variables:
        with open('objs.pkl', 'wb') as f:
            pickle.dump([self.chars, self.char_indices, self.indices_char,
                         self.sequence_length], f)


# Train the model:
autocc = Autocomplete()
features, labels = autocc.batchSampling()
autocc.trainLSTM_model(features, labels)
