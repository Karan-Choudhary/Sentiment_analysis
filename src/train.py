import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

from get_data import PreprocessData
from model import Model

# create a class for training the model which inherit Model class

class TrainModel(Model):
    def __init__(self, path):
        super().__init__()
        self.model = super().build_model()
        self.path = path
    
    def train_model(self):
        PreprocessDataClass = PreprocessData(self.path)
        X_train, X_test, y_train, y_test = PreprocessDataClass.preprocessing()
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
        
        # save the model
        self.model.save('saved_models/model.h5')

        return self.model