import tensorflow as tf
from tensorflow.keras.layers import Embedding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

# Create a model class to Initialize the model
class Model:
    def __init__(self):
        self.model = None
    
    def build_model(self):
        embedding_vector_features = 40
        model = Sequential()
        model.add(Embedding(5000, embedding_vector_features, input_length=20))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return self.model