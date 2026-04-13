from fastapi.testclient import TestClient
from app.main import app

# On crée un client virtuel pour attaquer notre API
client = TestClient(app)

# Un dictionnaire de test valide (ton assuré de 66 ans)
PAYLOAD_TEST = {
    "index": 50000,
    "bonus": 0.58,
    "type_contrat": "Maxi",
    "duree_contrat": 1,
    "anciennete_info": 1,
    "freq_paiement": "Yearly",
    "paiement": "No",
    "utilisation": "Retired",
    "code_postal": "28388",
    "conducteur2": "No",
    "age_conducteur1": 66,
    "age_conducteur2": 0,
    "sex_conducteur1": "F",
    "sex_conducteur2": "",
    "anciennete_permis1": 34,
    "anciennete_permis2": 0,
    "anciennete_vehicule": 16.0,
    "cylindre_vehicule": 1239,
    "din_vehicule": 55,
    "essence_vehicule": "Gasoline",
    "marque_vehicule": "RENAULT",
    "modele_vehicule": "CLIO",
    "debut_vente_vehicule": 16,
    "fin_vente_vehicule": 15,
    "vitesse_vehicule": 150,
    "type_vehicule": "Tourism",
    "prix_vehicule": 10321,
    "poids_vehicule": 830,
}


def test_health_check():
    """Vérifie que l'API est bien en vie."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "message": "healthy",
        "status": "healthy",
        "version": "1.0.0",
    }


def test_predict_freq_route():
    """Vérifie que la route de fréquence renvoie bien une probabilité."""
    response = client.post("/predict_freq", json=PAYLOAD_TEST)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "frequence" in data
    assert 0.0 <= data["frequence"] <= 1.0  # Une proba doit être entre 0 et 1 !


def test_predict_premium_route():
    """Vérifie que la route de prime totale fonctionne et renvoie des nombres cohérents."""
    response = client.post("/predict_premium", json=PAYLOAD_TEST)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "prime_totale" in data
    assert data["prime_totale"] >= 0  # Une prime ne peut pas être négative
