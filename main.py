import os
from fastapi import FastAPI, Header
import uvicorn
import logging
from joblib import load
import datetime as dt
logging.basicConfig(filename='logger_api.log', level=logging.INFO)


app = FastAPI()

@app.get("/test/")
def test():
    return {"Message" : "Bonjour, ceci est la beta d'un algorithme d'analyse de sentiment",
            "Status Code": 200}

@app.get("/request_logs/")        
def request_logs():
    with open('logger_api.log') as f:
        logs = f.read().splitlines()
    return logs

@app.post("/get_sentiment/")
def return_sentiment(comment: str):
        clf_pipe = load('updated_sentiment_classifier.joblib')
        prediction = clf_pipe.predict([comment])[0]
        prediction = "Positif" if prediction == 1 else "Négatif"
        logging.info(f'At {dt.datetime.now()}, predicted {prediction} from {comment}')
        return {"input text" : comment,
                "prediction":prediction,
                "Status code" : 200}