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

from get_data import PreprocessData
from model import Model
from train import TrainModel
from Evaluate import EvaluateModel




if __name__ == '__main__':
    # call classes
    path = 'data/airline_sentiment_analysis.csv'

    print("=====================================")
    # print("Training started...")
    
    
    # trainclass = TrainModel(path)
    # trainclass.train_model()
    
    # print("Training completed...")
    # print("=====================================")
    print("Evaluation started...")
    
    evalClass = EvaluateModel(path)
    evalClass.evaluate_model()
    print("Evaluation completed...")
    print("=====================================")
