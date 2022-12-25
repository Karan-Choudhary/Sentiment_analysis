from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

from tensorflow.keras.models import load_model

from app.src.get_data import PreprocessData
from app.src.train import TrainModel

# create a class for evaluating the model

class EvaluateModel(PreprocessData):
    '''
    This class is used to evaluate the model
    If model is not found, it will train the model and then evaluate it
    
    Performance metrics:
    1. Confusion matrix
    2. Classification report
    3. Accuracy score
    '''
    
    def __init__(self, path):
        super().__init__(path)
        self.path = path
        _, self.X_test, _, self.y_test = super().preprocessing()
        
        self.model = None
        
        try:
            self.model = load_model('saved_models/Final_model.h5')
        except:
            print("Model not found...")
            print("Starting model training phase...")
            trainClass = TrainModel(self.path)
            self.model = trainClass.train_model()
        
    def evaluate_model(self):
        print("=====================================")
        print("Evaluating the model...")
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        
        cm = confusion_matrix(self.y_test, y_pred)
        classification_result = classification_report(self.y_test, y_pred)
        
        print("Evaluating completed...")
        print("Results:")
        
        print(cm)
        print(classification_report(self.y_test, y_pred))
        print(accuracy_score(self.y_test, y_pred))
        print("=====================================")
        # write the result in a json file
        with open('results/result.json', 'w') as f:
            f.write(classification_result)
            f.write(str(cm))
            f.write(str(accuracy_score(self.y_test, y_pred)))
        
        f.close()   
        