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
        voc_size = 10000
        
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
        sent_length = 20
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        
        self.X_final = np.array(embedded_docs)
        self.y_final = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X_final, self.y_final, test_size=0.3, random_state=42, stratify=self.y_final)
        return X_train, X_test, y_train, y_test
    
    def preprocessing_for_prediction(self,dataset):
        voc_size = 10000
        
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
        sent_length = 20
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        
        self.X_final = np.array(embedded_docs)
        
        return self.X_final
        
    


# call classes
# data = PreprocessData('data/airline_tweets.csv')