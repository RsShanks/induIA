from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.services.service import predict_frequency, predict_severity

app = FastAPI(title="InduIA - Pricing API")


@app.get("/")
async def root():
    return FileResponse("app/templates/index.html")


@app.get("/health")
async def health_check():
    return {"message": "healthy", "status": "healthy", "version": "1.0.0"}


# --- Route Fréquence ---
@app.post("/predict_freq")
async def get_predict_freq(data: dict):
    proba = predict_frequency(data)
    return {"status": "success", "frequence": proba}


# --- Route Sévérité ---
@app.post("/predict_sev")
async def get_predict_sev(data: dict):
    montant = predict_severity(data)
    return {"status": "success", "severite": montant}


# --- Route Prime Totale (Fréquence * Sévérité) ---
@app.post("/predict_premium")
async def get_predict_premium(data: dict):
    proba = predict_frequency(data)
    montant = predict_severity(data)
    prime_pure = proba * montant

    return {
        "status": "success",
        "details": {"frequence": proba, "severite": montant},
        "prime_totale": prime_pure,
    }
