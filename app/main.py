from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
from typing import Dict, Any, Optional
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
    description="API de tarification d'assurance automobile",
    version="1.0.0",
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
    Agit comme un bouclier pour garantir la cohérence des données actuarielles.
    """

    model_config = ConfigDict(extra="allow")

    # Contrat & Historique
    type_contrat: str
    duree_contrat: int = Field(ge=1, description="Durée du contrat en mois")
    anciennete_info: int
    freq_paiement: str
    paiement: str
    utilisation: str
    code_postal: str

    # Conducteur Principal
    age_conducteur1: int = Field(ge=16, description="Âge du conducteur principal")
    sex_conducteur1: str
    anciennete_permis1: int = Field(ge=0)
    bonus: float

    # Second Conducteur
    conducteur2: str
    age_conducteur2: int = Field(ge=0, description="Âge du conducteur secondaire")
    sex_conducteur2: Optional[str] = ""
    anciennete_permis2: int = Field(
        ge=0, description="Années de permis du conducteur secondaire"
    )

    # Véhicule
    marque_vehicule: str
    modele_vehicule: str
    type_vehicule: str
    prix_vehicule: float = Field(gt=0)
    anciennete_vehicule: float
    poids_vehicule: int
    vitesse_vehicule: int
    essence_vehicule: str
    din_vehicule: int
    cylindre_vehicule: int

    @model_validator(mode="after")
    def check_coherence_age_permis(self) -> Self:
        age_minimum_obtention = 15

        # --- 1. Vérification du Conducteur Principal ---
        anciennete_max_possible1 = self.age_conducteur1 - age_minimum_obtention
        if self.anciennete_permis1 > anciennete_max_possible1:
            raise ValueError(
                f"Incohérence : Un conducteur principal de {self.age_conducteur1} ans "
                f"ne peut pas avoir {self.anciennete_permis1} ans de permis."
            )

        # --- 2. Vérification du Conducteur Secondaire (S'il existe) ---
        if self.conducteur2 == "Yes":
            if self.age_conducteur2 < 16:
                raise ValueError("Le deuxième conducteur doit avoir au moins 16 ans.")

            anciennete_max_possible2 = self.age_conducteur2 - age_minimum_obtention
            if self.anciennete_permis2 > anciennete_max_possible2:
                raise ValueError(
                    f"Incohérence : Un conducteur secondaire de {self.age_conducteur2} ans "
                    f"ne peut pas avoir {self.anciennete_permis2} ans de permis."
                )

        # --- 3. Règles Métier ---
        if self.bonus < 0 or self.bonus > 3.5:
            raise ValueError("Le bonus/malus (CRM) doit être compris entre 0 et 3.5.")

        if self.utilisation == "Retired" and self.age_conducteur1 < 50:
            raise ValueError("Un conducteur retraité doit avoir au moins 50 ans.")

        return self


# =====================================================================
# --- 3. ROUTES DE L'API ---
# =====================================================================


@app.get("/")
async def root() -> FileResponse:
    """Sert la page web principale (Interface Utilisateur)."""
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
        data = client.model_dump()

        proba = engine.predict_frequency(data)
        montant = engine.predict_severity(data)
        alpha = engine.get_alpha()

        prime_pure = proba * montant * alpha

        return {
            "status": "success",
            "details": {
                "frequence": round(proba, 5),
                "severite": round(montant, 2),
                "alpha_calibration": alpha,
            },
            "prime_totale": round(prime_pure, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
