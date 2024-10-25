import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from BankNotes import BankNote


app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))
@app.get('/')
def index():
    return {'Message' : 'FastAPI implementation in ML model'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]
    prediction = model.predict([[variance, skewness, curtosis, entropy]])[0]
    if(prediction==0):
        pred = "valid note"
    else:
        pred = "Invalid note"
    return {'model prediction' : pred}

if __name__ == "__main__":
    uvicorn.run(app)