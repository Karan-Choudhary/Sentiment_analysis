from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
# from mangum import Mangum
# import uvicorn

from app.inference import Inference

app = FastAPI()
# handler = Mangum(app)

class TextIn(BaseModel):
    text: str
    
class PredictionOut(BaseModel):
    sentiment: str

@app.post("/predict/{message}", response_model=PredictionOut)
def predict(message: str):
    # print the message
    message_list = [message]
    # convert the message into a dataset
    inferenceClass = Inference()
    inferenceClass.convert_into_dataset(message_list)
    # make predictions
    y_pred = inferenceClass.predict()
    
    # return the result
    
    if y_pred[0][0] == 0:
        sentiment = "Negative"
        # return {"prediction": "Negative"}
        return {"sentiment": sentiment}
    else:
        sentiment = "Positive"
        # return {"prediction": "Positive"}
        return {"sentiment": sentiment}
    
    sentiment = "Something went wrong"
    return {"sentiment": sentiment}
    # return {"prediction": "Something went wrong"}
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9000)