import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from category_encoders import TargetEncoder
import optuna
import pickle


# Rendre Optuna silencieux
optuna.logging.set_verbosity(optuna.logging.WARNING)

# df = pd.read_csv('train.csv')
# categorical_features = ['type_contrat', 'utilisation', 'sex_conducteur1', 'essence_vehicule'] 
# numerical_features = ['bonus', 'anciennete_vehicule', 'age_conducteur1','puissance_age','ratio_poids_puissance', 'log_prix_vehicule', 'cp_dep', 'ratio_permis_age','din_vehicule']

def data():
    df = pd.read_csv('data/training/train.csv')
    return df

def cat():
    categorical_features = ['type_contrat', 'utilisation', 'sex_conducteur1', 'essence_vehicule'] 
    numerical_features = ['bonus', 'anciennete_vehicule', 'age_conducteur1','puissance_age','ratio_poids_puissance', 'log_prix_vehicule', 'cp_dep', 'ratio_permis_age','din_vehicule']
    return categorical_features, numerical_features
#####################
# --- FONCTION DE NETTOYAGE ---
def clean_data():
    df = data()
    df['montant_sinistre'] = df['montant_sinistre'].fillna(0)
    df['montant_sinistre'] = df['montant_sinistre'].where(df['montant_sinistre'] >= 0, 0)
    df = df[df['poids_vehicule'] > 0]

    return df

# --- 1. FONCTION DE FEATURE ENGINEERING ---
def apply_feature_engineering(df):
    df['ratio_poids_puissance'] = df['poids_vehicule'] / (df['cylindre_vehicule'] + 1)
    df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])     
    df['cp_dep'] = df['code_postal'].astype(str).str.zfill(5).str[:2]
    df['is_sportive'] = ((df['din_vehicule'] > 150) | (df['vitesse_vehicule'] > 200)).astype(int)
    df["ratio_permis_age"] = df["anciennete_permis1"] / (df["age_conducteur1"] + 1)
    df["puissance_age"] = df["din_vehicule"] * df["age_conducteur1"]
    return df

# Application
# df = clean_data(df)
# df = apply_feature_engineering(df)

def df_clean_fe():
    df = clean_data()
    df = apply_feature_engineering(df)
    return df


# --- b. Préparation ---

def prepare_data():
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X = df[numerical_features + categorical_features]
    y_freq = df['nombre_sinistres']
    X_train, X_test, y_train, y_test = train_test_split(X, y_freq, test_size=0.2, random_state=8)
    encoder = TargetEncoder(cols=['cp_dep'])
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)
    return X_train, X_test, y_train, y_test,categorical_features


# # =====================================================================
# # --- 2. MODÈLE FRÉQUENCE (Probabilité de sinistre) ---
# # =====================================================================


# --- b. Préparation ---
##optuna
def objective_freq(trial):
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X = df[numerical_features + categorical_features]
    y_freq = df['nombre_sinistres']
    X_train, X_test, y_train, y_test,categorical_features = prepare_data()
    params = {
        "iterations": trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "eval_metric": "AUC",
        "verbose": False
    }
    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    model = CatBoostClassifier(**params, auto_class_weights="Balanced")
    model.fit(X_tr_opt, y_tr_opt, cat_features=categorical_features, eval_set=(X_val_opt, y_val_opt), early_stopping_rounds=20)    
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, preds_proba)
    
    print(f'Freq parameters: {params}, AUC: {auc_val:.4f}')
    return auc_val

########
def train_model_freq():
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X = df[numerical_features + categorical_features]
    y_freq = df['nombre_sinistres']
    X_train, X_test, y_train, y_test,categorical_features = prepare_data()
    print("--- Recherche des hyperparamètres FRÉQUENCE (Maximisation AUC) ---")
    study_freq = optuna.create_study(direction="maximize")
    study_freq.optimize(objective_freq, n_trials=10) 

    # Entraînement final Fréquence
    print(f"\nBest Freq parameters: {study_freq.best_params}")
    model_freq = CatBoostClassifier(**study_freq.best_params, auto_class_weights="Balanced", eval_metric="AUC", verbose=False)
    model_freq.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_test, y_test))
    #sauvegarde catboost
    model_freq.save_model('data/models/model_freq.cbm')
    

    return model_freq
# =====================================================================
# --- 3. MODÈLE SÉVÉRITÉ (Montant moyen) ---
# =====================================================================
def model_severite():
    df = df_clean_fe()
    encoder = TargetEncoder(cols=['cp_dep'])
    categorical_features, numerical_features = cat()
    df_claims = df.copy() 
    X_sev = df_claims[numerical_features + categorical_features]
    y_sev = df_claims['montant_sinistre']
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_sev, y_sev, test_size=0.2, random_state=8)
    X_train_sev = encoder.fit_transform(X_train_sev, y_train_sev)
    X_test_sev = encoder.transform(X_test_sev)
    return X_train_sev, X_test_sev, y_train_sev, y_test_sev,categorical_features

def objective_sev(trial):
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X_train_sev, X_test_sev, y_train_sev, y_test_sev,categorical_features=model_severite()         
    params = {
        "iterations": trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "objective": 'Tweedie:variance_power=1.5', 
        "verbose": False
    }
    
    model = CatBoostRegressor(**params) # Pas de class_weights pour une régression
    X_tr_sev_opt, X_val_sev_opt, y_tr_sev_opt, y_val_sev_opt = train_test_split(
        X_train_sev, y_train_sev, test_size=0.2, random_state=42
    )
    model.fit(X_tr_sev_opt, y_tr_sev_opt, cat_features=categorical_features, eval_set=(X_val_sev_opt, y_val_sev_opt), early_stopping_rounds=20)    
    preds = model.predict(X_test_sev)
    mse_val = mean_squared_error(y_test_sev, preds)
    
    print(f'Sev parameters: {params}, MSE: {mse_val:.2f}')
    return mse_val


def train_model_sev():
    df = df_clean_fe()
    categorical_features, numerical_features = cat()
    X = df[numerical_features + categorical_features]
    y_freq = df['nombre_sinistres']
    X_train_sev, X_test_sev, y_train_sev, y_test_sev,categorical_features = prepare_data()
    print("\n--- Recherche des hyperparamètres SÉVÉRITÉ (Minimisation MSE) ---")
    study_sev = optuna.create_study(direction="minimize")
    study_sev.optimize(objective_sev, n_trials=10) # Ajuster si besoin

    # Entraînement final Sévérité
    print(f"\nBest Sev parameters: {study_sev.best_params}")
    # On s'assure de réinjecter l'objectif Tweedie qui n'est pas dans best_params par défaut s'il n'était pas dans l'espace de recherche suggest
    best_sev_params = study_sev.best_params
    best_sev_params['objective'] = 'Tweedie:variance_power=1.5'

    model_sev = CatBoostRegressor(**best_sev_params, verbose=False)
    model_sev.fit(X_train_sev, y_train_sev, cat_features=categorical_features, eval_set=(X_test_sev, y_test_sev))
    model_sev.save_model('data/models/model_sev.cbm')
    print("Modèles exportés avec succès !")
    return model_sev


# # =====================================================================
# # --- MÉTRIQUES FINALES ---
# # =====================================================================
# def predict_final():
#     df = df_clean_fe()
#     categorical_features, numerical_features = cat()
#     X = df[numerical_features + categorical_features]
#     y_freq = df['nombre_sinistres']
#     X_test, y_test, X_test_sev, y_test_sev,categorical_features = prepare_data()
#     model_freq = train_model_freq()
#     model_sev = train_model_sev()
#     encoder = TargetEncoder(cols=['cp_dep'])
#     y_pred_freq_proba = model_freq.predict_proba(X_test)[:, 1]
#     auc_final = roc_auc_score(y_test, y_pred_freq_proba)
#     print(f"\nROC AUC final for the frequency model: {auc_final:.4f}")

#     y_pred_sev = model_sev.predict(X_test_sev)
#     mse_sev_final = mean_squared_error(y_test_sev, y_pred_sev)
#     print(f"Mean Squared Error final for the severity model: {mse_sev_final:.2f}")


#     # --- 4. CALCUL DU COÛT ESPÉRÉ SUR TOUT LE DATASET ---
#     X_encoded = encoder.transform(X)

#     df['probabilite'] = model_freq.predict_proba(X_encoded)[:, 1]
#     df['montant_potentiel'] = model_sev.predict(X_encoded)
#     df['pred'] = df['probabilite'] * df['montant_potentiel']

#     print("\nAperçu des prédictions :")
#     print(df[['probabilite', 'montant_potentiel', 'pred']].head())

