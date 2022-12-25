import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


class PreprocessData:
    '''
    This class is used for loading the data and preprocessing the data
    Preprocessing includes:
    1. Removing special characters
    2. Removing stopwords
    3. Stemming
    4. One hot encoding
    5. Padding
    6. Embedding
    7. Splitting the data into train and test
    '''
    def __init__(self, path):
        self.path = path
        
        self.data = pd.DataFrame()
        
        self.X_final = None
        self.y_final = None
        nltk.download('stopwords')
    
    def load_data(self):
        self.data = pd.read_csv(self.path)
        return self.data
    
    def splitting(self):
        self.data = PreprocessData.load_data(self)
        self.data['label'] = self.data['airline_sentiment'].map({'positive':1, 'negative':0})
        X = self.data['text']
        y = self.data['label']
        return X, y
    
    def preprocessing(self):
        
        print("=====================================")
        print("Loading data and preprocessing...")
        
        # voc_size = 10000
        
        voc_size = 90000
        
        X, y = PreprocessData.splitting(self)
        
        messages = X.copy()
        ps = PorterStemmer()
        corpus = []
        
        for i in range(len(messages)):
            review = re.sub('[^a-zA-Z]', ' ', messages[i])
            review = review.lower()
            review = review.split()
            
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        
        onehot_repr = [one_hot(words, voc_size) for words in corpus]
        # sent_length = 20
        sent_length = 40
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        
        self.X_final = np.array(embedded_docs)
        self.y_final = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X_final, self.y_final, test_size=0.3, random_state=42, stratify=self.y_final)
        
        print("Data loaded and preprocessed...")
        print("=====================================")
        return X_train, X_test, y_train, y_test
    
    def preprocessing_for_prediction(self,dataset):
        print("=====================================")
        print("Loading data and preprocessing...")
        
        # voc_size = 10000
        voc_size = 90000
        
        messages = dataset['text'].copy()
        ps = PorterStemmer()
        corpus = []
        
        for i in range(len(messages)):
            review = re.sub('[^a-zA-Z]', ' ', messages[i])
            review = review.lower()
            review = review.split()
            
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        
        onehot_repr = [one_hot(words, voc_size) for words in corpus]
        # sent_length = 20
        sent_length = 40
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        
        self.X_final = np.array(embedded_docs)
        print("Data loaded and preprocessed...")
        print("=====================================")
        
        return self.X_final
