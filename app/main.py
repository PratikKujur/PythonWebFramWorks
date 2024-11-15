from  fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_pipleline
from model.predict import __version__ as model_version

app=FastAPI()

class TextIn(BaseModel):
    text:str

class PredictionOut(BaseModel):
    language:str

@app.get("/")
def home():
    return {"health_check":"OK","model_version":model_version}

@app.post("/predict",response_model=PredictionOut)
def predict(payload:TextIn):
    language=predict_pipleline(payload.text)
    return {"language" : language}