from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

from app.inference import Inference

app = FastAPI()

class TextIn(BaseModel):
    text: str
    
class PredictionOut(BaseModel):
    sentiment: str

@app.post("/predict/{message}", response_model=PredictionOut)
def predict(message: str):
    message_list = [message]
    inferenceClass = Inference()
    inferenceClass.convert_into_dataset(message_list)
    
    y_pred = inferenceClass.predict()
    
    
    if y_pred[0][0] == 0:
        sentiment = "Negative"
        return {"sentiment": sentiment}
    else:
        sentiment = "Positive"
        return {"sentiment": sentiment}
    
    sentiment = "Something went wrong"
    return {"sentiment": sentiment}