from fastapi import FastAPI
from app.services.service import train_model_freq, train_model_sev
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
gitlfs
svg catbost via catboost

@app.get("/predict/{input_data}")
async def predict(input_data: str):
    # Here you would load your trained model and make a prediction based on the input_data
    # For example:
    # model = load_model('path_to_your_model')
    # prediction = model.predict(input_data)
    # return {"prediction": prediction}
    return {"message": f"Received input data: {input_data}. Prediction functionality not implemented yet."}


class PricingInput(BaseModel):
    # Informations Conducteur
    age_conducteur1: int = Field(..., ge=18, le=100)
    anciennete_permis1: int = Field(..., ge=0)
    sex_conducteur1: str # ex: 'M' ou 'F'
    bonus: float = Field(..., ge=0.5, le=3.5)
    
    # Informations Véhicule
    poids_vehicule: float = Field(..., gt=0)
    cylindre_vehicule: int = Field(..., ge=0)
    prix_vehicule: float = Field(..., gt=0)
    din_vehicule: int = Field(..., gt=0)
    vitesse_vehicule: int = Field(..., gt=0)
    anciennete_vehicule: int = Field(..., ge=0)
    essence_vehicule: str # ex: 'Essence', 'Diesel', 'Electrique'
    
    # Contrat et Localisation
    code_postal: str = Field(..., min_length=5, max_length=5)
    type_contrat: str
    utilisation: str # ex: 'Privé', 'Pro', 'Trajet-Travail'

    # Exemple de validateur pour le code postal
    @validator('code_postal')
    def validate_cp(cls, v):
        if not v.isdigit():
            raise ValueError('Le code postal doit contenir uniquement des chiffres')
        return v

# 2. On utilise POST
@app.post("/predict")
async def predict_premium(data: PricingInput):
    # Ici, 'data' est déjà validé et typé !
    # On peut accéder aux valeurs : data.age, data.bonus_malus
    tarif = 500 * data.bonus_malus
    # if data.age < 25:
    #     tarif += 200
        
    return {"status": "success", "estimated_premium": tarif}