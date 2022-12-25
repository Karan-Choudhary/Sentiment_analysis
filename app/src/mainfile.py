import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

from app.src.get_data import PreprocessData
from app.src.model import Model
from app.src.train import TrainModel
from app.src.Evaluate import EvaluateModel


if __name__ == '__main__':
    '''
    This is the main file which calls all the classes
    From this file, user can select the option to train and evaluate the model or evaluate the model only
    '''
    # call classes
    # please change the path of the dataset accordingly
    path = 'data/airline_sentiment_analysis.csv'
    
    print("=====================================")
    print("Please select the option:")
    print("1. Train and evaluate the model")
    print("2. Evaluate the model only")
    print("=====================================")
    
    # Take input from the user
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        # Train the model
        trainclass = TrainModel(path)
        trainclass.train_model()
        
        # Evaluate the model
        evalClass = EvaluateModel(path)
        evalClass.evaluate_model()
    
    elif choice == 2:
        evalClass = EvaluateModel(path)
        evalClass.evaluate_model()
