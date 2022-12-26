import pandas
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


class PrepareData:
    def __init__(self):
        self.X_final = None
        nltk.download('stopwords')
    
    def preprocess(self, dataset):
        print("=====================================")
        print("Loading data and preprocessing...")
        
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
        sent_length = 40
        embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
        
        self.X_final = np.array(embedded_docs)
        print("Data loaded and preprocessed...")
        print("=====================================")
        
        return self.X_final