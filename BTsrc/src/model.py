import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.layers import Activation

# Create a model class to Initialize the model
class Model:
    '''
    This class is used to initialize the model
    Model is initialized with the following parameters:
    1. vocab_size: size of the vocabulary
    2. dim: dimension
    3. embedding_matrix: embedding matrix
    4. max_length: maximum length of the sentence
    '''
    
    def __init__(self):
        np.random.seed(1337)
        self.model = None
        self.vocab_size = 90000
        self.dim = 200
        self.embedding_matrix = np.random.randn(self.vocab_size + 1, self.dim) * 0.01
        self.max_length = 40
    
    # def build_model(self):
    #     embedding_vector_features = 40
    #     model = Sequential()
    #     model.add(Embedding(5000, embedding_vector_features, input_length=20))
    #     model.add(Dropout(0.3))
    #     model.add(Bidirectional(LSTM(100)))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(1, activation='sigmoid'))
    #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     self.model = model
    #     return self.model
    
    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size +1, self.dim, weights = [self.embedding_matrix], input_length = self.max_length))
        model.add(Dropout(0.4))
        model.add(LSTM(128))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self.model