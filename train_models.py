import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
df = pd.read_csv('train.csv')

# --- 1. Préparation ---
# On garde les colonnes telles quelles (CatBoost gère les strings)
categorical_features = ['type_contrat', 'utilisation', 'essence_vehicule', 'code_postal', 'marque_vehicule']
numerical_features = ['bonus', 'duree_contrat', 'age_conducteur1', 'anciennete_permis1', 'prix_vehicule', 
                      'vitesse_vehicule', 'cylindre_vehicule', 'anciennete_vehicule', 'din_vehicule', 'poids_vehicule']

# Remplissage simple pour les numériques (CatBoost gère aussi les NaNs, mais c'est plus propre)
df['anciennete_vehicule'] = df['anciennete_vehicule'].fillna(-1)

# --- 2. MODÈLE FRÉQUENCE (Probabilité de sinistre) ---
X = df[numerical_features + categorical_features]
y_freq = (df['nombre_sinistres'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_freq, test_size=0.2, random_state=8)

model_freq = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=categorical_features, # On lui donne directement la liste
    verbose=100
)

model_freq.fit(X_train, y_train, eval_set=(X_test, y_test))

# --- 3. MODÈLE SÉVÉRITÉ (Montant moyen) ---
# Uniquement sur les lignes avec un montant > 0
df_claims = df[df['montant_sinistre'] > 0].copy()
X_sev = df_claims[numerical_features + categorical_features]
y_sev = df_claims['montant_sinistre']

model_sev = CatBoostRegressor(
    iterations=500,
    objective='Tweedie:variance_power=1.5', # Très efficace pour les montants d'assurance
    cat_features=categorical_features,
    verbose=100
)

model_sev.fit(X_sev, y_sev)


# Sauvegarde des modèles
with open('model_freq.pkl', 'wb') as f:
    pickle.dump(model_freq, f)

with open('model_sev.pkl', 'wb') as f:
    pickle.dump(model_sev, f)

# --- 4. CALCUL DU COÛT ESPÉRÉ ---
df['probabilite'] = model_freq.predict_proba(X)[:, 1]
df['montant_potentiel'] = model_sev.predict(X)
df['cout_espere'] = df['probabilite'] * df['montant_potentiel']

