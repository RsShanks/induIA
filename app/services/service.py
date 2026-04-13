import os
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from category_encoders import TargetEncoder
import optuna

# Rendre Optuna silencieux
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# --- FONCTIONS UTILITAIRES DE BASE ---
# =====================================================================

def data():
    return pd.read_csv('data/training/train.csv')

def cat():
    categorical_features = ['type_contrat', 'utilisation', 'sex_conducteur1', 'essence_vehicule'] 
    numerical_features = ['bonus', 'anciennete_vehicule', 'age_conducteur1','puissance_age','ratio_poids_puissance', 'log_prix_vehicule', 'cp_dep', 'ratio_permis_age','din_vehicule']
    return categorical_features, numerical_features

def clean_data():
    df = data()
    df['montant_sinistre'] = df['montant_sinistre'].fillna(0)
    df['montant_sinistre'] = df['montant_sinistre'].where(df['montant_sinistre'] >= 0, 0)
    df = df[df['poids_vehicule'] > 0]
    return df

def apply_feature_engineering(df):
    df['ratio_poids_puissance'] = df['poids_vehicule'] / (df['cylindre_vehicule'] + 1)
    df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])     
    df['cp_dep'] = df['code_postal'].astype(str).str.zfill(5).str[:2]
    df['is_sportive'] = ((df['din_vehicule'] > 150) | (df['vitesse_vehicule'] > 200)).astype(int)
    df["ratio_permis_age"] = df["anciennete_permis1"] / (df["age_conducteur1"] + 1)
    df["puissance_age"] = df["din_vehicule"] * df["age_conducteur1"]
    return df

def df_clean_fe():
    df = clean_data()
    df = apply_feature_engineering(df)
    return df

def prepare_data():
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X = df[numerical_features + categorical_features]
    y_freq = df['nombre_sinistres']
    X_train, X_test, y_train, y_test = train_test_split(X, y_freq, test_size=0.2, random_state=8)
    
    # Création et sauvegarde de l'encodeur global pour 'cp_dep'
    os.makedirs('data/models', exist_ok=True)
    encoder = TargetEncoder(cols=['cp_dep'])
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)
    
    with open('data/models/target_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
        
    return X_train, X_test, y_train, y_test, categorical_features


# =====================================================================
# --- FONCTIONS D'INFÉRENCE POUR L'API (Endpoints) ---
# =====================================================================


def predict_frequency(data_dict: dict) -> float:
    # 1. Préparation des données
    df_input = pd.DataFrame([data_dict])
    df_processed = apply_feature_engineering(df_input)
    
    categorical_features, numerical_features = cat()
    X_pred = df_processed[numerical_features + categorical_features]

    # 2. On charge le fichier pkl et on transforme (LE FAMEUX DICTIONNAIRE)
    with open('data/models/target_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    X_pred_encoded = encoder.transform(X_pred)

    # 3. On charge le modèle cbm et on prédit (LE 2ÈME DICTIONNAIRE)
    model_freq = CatBoostClassifier()
    model_freq.load_model('data/models/model_freq.cbm')
    proba_sinistre = model_freq.predict_proba(X_pred_encoded)[0][1]
    
    return float(proba_sinistre)


def predict_severity(data_dict: dict) -> float:
    # 1. Préparation des données (comme pour la fréquence)
    df_input = pd.DataFrame([data_dict])
    df_processed = apply_feature_engineering(df_input)
    
    categorical_features, numerical_features = cat()
    X_pred = df_processed[numerical_features + categorical_features]

    # 2. On charge l'encodeur et on transforme (SURTOUT PAS de .fit() ici)
    with open('data/models/target_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    X_pred_encoded = encoder.transform(X_pred)

    # 3. On charge le modèle de sévérité (CatBoostRegressor)
    model_sev = CatBoostRegressor()
    model_sev.load_model('data/models/model_sev.cbm')
    
    # 4. On fait la prédiction
    montant_estime = model_sev.predict(X_pred_encoded)[0]
    
    # Règle métier : un sinistre ne peut pas coûter un montant négatif
    return max(0.0, float(montant_estime))