from fastapi import FastAPI
from app.services.service import train_model_freq, train_model_sev, predict_frequency, predict_severity
from pydantic import BaseModel, Field, validator

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"message": "healthy","status": "healthy", "version": "1.0.0"}

@app.get("/train_model/{model}")
async def train_models(model: str):
    
    if model == 'freq':
        train_model_freq()
        return {"message": "Model 'frequency' trained and saved successfully."}
    
    if model == 'sev':
        train_model_sev()
        return {"message": "Model 'severity' trained and saved successfully."}

# 2. On utilise POST
@app.post("/predict_freq")
async def predict_premium(data: dict):
    proba = predict_frequency(data)
    return {"status": "success", "data": proba}


@app.post("/predict_sev")
async def predict_premium(data: dict):
    montant = predict_severity(data)
    return {"status": "success", "data": montant}