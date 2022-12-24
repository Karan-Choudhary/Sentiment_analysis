from typing import Union
from fastapi import FastAPI

from src.inference import Inference

# create a fastapi app for serving the model as an API and make predictions
app = FastAPI()

@app.post("/predict/{message}")
def predict(message: str):
    # print the message
    message_list = [message]
    # convert the message into a dataset
    inferenceClass = Inference()
    inferenceClass.convert_into_dataset(message_list)
    # make predictions
    y_pred = inferenceClass.predict()
    
    print("=====================================")
    print(y_pred)
    print(type(y_pred))
    print(y_pred[0][0])
    print("=====================================")
    # return the result
    
    if y_pred[0][0] == 0:
        return {"prediction": "Negative"}
    else:
        return {"prediction": "Positive"}
    
    return {"prediction": message}