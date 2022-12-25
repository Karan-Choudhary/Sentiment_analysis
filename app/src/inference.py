import pandas as pd
from tensorflow.keras.models import load_model
from app.src.get_data import PreprocessData


# inference class
class Inference(PreprocessData):
    def __init__(self):
        self.model = load_model('app/saved_models/Final_model.h5')
        self.dataset = None
        self.X_final = None
        
    def convert_into_dataset(self, messages):
        self.dataset = pd.DataFrame(messages, columns=['text'])
        PreprocessDataClass = PreprocessData("")
        self.X_final = PreprocessDataClass.preprocessing_for_prediction(self.dataset)
        
    def predict(self):
        y_pred = (self.model.predict(self.X_final) > 0.5).astype("int32")
        return y_pred