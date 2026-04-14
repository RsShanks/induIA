import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
from app.logging_config import setup_logging
from app.services.service import TarificationEngine
from fastapi.staticfiles import StaticFiles


setup_logging()

engine = TarificationEngine()
# =====================================================================
# --- 1. INITIALISATION DE L'APPLICATION ET DU MOTEUR ---
# =====================================================================

app = FastAPI(
    title="InduIA - Pricing API",
    description="API de tarification d'assurance automobile avec Machine Learning",
    version="1.0.0"
)
app.mount("/logo", StaticFiles(directory="app/logo"), name="logo")
# On allume le moteur (qui va charger les fichiers .pkl en mémoire 1 seule fois)
engine = TarificationEngine()


# =====================================================================
# --- 2. MODÈLES DE VALIDATION DES DONNÉES (PYDANTIC) ---
# =====================================================================

class ClientInput(BaseModel):
    """
    Modèle de validation des requêtes entrantes.
    Vérifie le typage des variables critiques avant de les envoyer au moteur.
    """
    model_config = ConfigDict(extra='allow') 
    
    age_conducteur1: int
    anciennete_permis1: int
    prix_vehicule: float
    duree_contrat: int


# =====================================================================
# --- 3. ROUTES DE L'API ---
# =====================================================================

@app.get("/")
async def root() -> FileResponse:
    """Sert la page web principale (Interface Utilisateur)."""
    # On utilise le chemin exact que tu avais paramétré
    return FileResponse("app/templates/index.html")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Route de vérification pour s'assurer que le serveur est en ligne."""
    return {"message": "healthy", "status": "healthy", "version": "1.0.0"}


@app.post("/predict_freq")
async def get_predict_freq(client: ClientInput) -> Dict[str, Any]:
    """
    Calcule uniquement la fréquence (probabilité) de sinistre.
    """
    try:
        data = client.model_dump()
        proba = engine.predict_frequency(data)
        return {"status": "success", "frequence": round(proba, 5)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_sev")
async def get_predict_sev(client: ClientInput) -> Dict[str, Any]:
    """
    Calcule uniquement la sévérité (coût estimé) du sinistre.
    """
    try:
        data = client.model_dump()
        montant = engine.predict_severity(data)
        return {"status": "success", "severite": round(montant, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_premium")
async def get_predict_premium(client: ClientInput) -> Dict[str, Any]:
    """
    Calcule la prime pure totale en combinant fréquence, sévérité et calibration (alpha).
    C'est la route appelée par le formulaire HTML.
    """
    try:
        # 1. On transforme la requête validée en dictionnaire python
        data = client.model_dump()
        
        # 2. On utilise le moteur pour faire les prédictions
        proba = engine.predict_frequency(data)
        montant = engine.predict_severity(data)
        alpha = engine.get_alpha()
        
        # 3. Calcul de la prime pure
        prime_pure = proba * montant * alpha

        # 4. On renvoie le résultat formaté
        return {
            "status": "success",
            "details": {
                "frequence": round(proba, 5),
                "severite": round(montant, 2),
                "alpha_calibration": alpha
            },
            "prime_totale": round(prime_pure, 2),
        }
    except Exception as e:
        # S'il y a la moindre erreur de code, elle remontera en erreur HTTP 500
        raise HTTPException(status_code=500, detail=str(e))