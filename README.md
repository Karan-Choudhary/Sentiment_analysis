## Sentiment Analysis using LSTM
* [Try out the application](http://54.95.205.156:5000/docs)
* Sentiment Analysis is a process of determining whether a piece of writing is positive or negative.
* API is developed using [FastAPI](https://fastapi.tiangolo.com/)
* Model is build using [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
* Docker is used for containerization.

### Requirements
* [Create a new virtual environment](https://docs.python.org/3/library/venv.html).
* Python 3.6 or greater.
* Run Command-
```
pip install -r requirements.txt
```

### Usage:
* For Training the model, run:
```
BuildTrain.bat (For Windows)
``` 
* Put you own data in ***BTsrc/data*** directory named as airline_sentiment_analysis.csv
* Saved model will be saved in ***BTsrc/app/saved_models*** directory or you can download [Pretrained Model](https://githubmodels.s3.ap-northeast-1.amazonaws.com/Final_model.h5) from here and put in ***BTsrc/app/saved_models*** directory.
* To test the api, build the docker image using command:
```
docker build -t sentiment_analysis .
```
* Run the docker image using command:
```
docker run -p 80:80 sentiment_analysis
```
