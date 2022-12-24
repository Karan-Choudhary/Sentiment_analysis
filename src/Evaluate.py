from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

from tensorflow.keras.models import load_model

from get_data import PreprocessData

# create a class for evaluating the model

class EvaluateModel(PreprocessData):
    def __init__(self, path):
        super().__init__(path)
        _, self.X_test, _, self.y_test = super().preprocessing()
        self.model = load_model('saved_models\Final_model.h5')
        
    def evaluate_model(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        
        cm = confusion_matrix(self.y_test, y_pred)
        classification_result = classification_report(self.y_test, y_pred)
        
        print(cm)
        print(classification_report(self.y_test, y_pred))
        print(accuracy_score(self.y_test, y_pred))
        
        # write the result in a json file
        with open('results/result.json', 'w') as f:
            f.write(classification_result)
            f.write(str(cm))
            f.write(str(accuracy_score(self.y_test, y_pred)))
        
        f.close()   
        