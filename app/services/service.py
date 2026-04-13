import pandas as pd
import numpy as np
import pickle

# =====================================================================
# --- 1. CHARGEMENT DES BUNDLES EN MÉMOIRE ---
# =====================================================================

_FREQ_BUNDLE = None
_SEV_BUNDLE = None


def _load_bundles():
    """Charge les bundles (Modèle + Préprocesseur + Hyperparamètres) une seule fois."""
    global _FREQ_BUNDLE, _SEV_BUNDLE
    if _FREQ_BUNDLE is None:
        try:
            # On adapte les chemins vers ton dossier data/models/
            with open("app/data/models/model_frequence.pkl", "rb") as f:
                _FREQ_BUNDLE = pickle.load(f)
            with open("app/data/models/model_severite.pkl", "rb") as f:
                _SEV_BUNDLE = pickle.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Fichier modèle .pkl introuvable dans data/models/. Erreur: {e}"
            )


# =====================================================================
# --- 2. LOGIQUE DE TRANSFORMATION (Code de ta collaboratrice) ---
# =====================================================================


def apply_feature_engineering(df):
    """Reproduit le Feature Engineering de testmod.py"""
    df = df.copy()

    df["permis_par_age"] = df["anciennete_permis1"] / (df["age_conducteur1"] + 1)
    df["log_prix"] = np.log1p(df["prix_vehicule"])
    df["vehicule_puissant"] = (df["din_vehicule"] > 120).astype(int)
    df["vitesse_clip"] = df["vitesse_vehicule"].clip(50, 220)
    df["vitesse_vehicule"] = df["vitesse_vehicule"].clip(10, 250)
    df["exposition"] = (df["duree_contrat"] / 12).clip(lower=0.25)
    return df


def apply_te_preprocessor(df, prep):
    """Reproduit la fonction de transformation Target Encoding de testmod.py"""
    df = df.copy()

    # 1) Catégories -> Target Encoding
    for col in prep["cat_cols"]:
        if col not in df.columns:
            df[col] = prep["global_mean"]
            continue

        col_ser = df[col].fillna("MISSING").astype(str)
        if col in prep["keep_sets"]:
            keep = prep["keep_sets"][col]
            col_ser = col_ser.apply(lambda x: x if x in keep else "RARE")

        mapping = prep["te_maps"][col]
        df[col] = col_ser.map(mapping).fillna(prep["global_mean"]).astype(float)

    # 2) Numériques -> Remplissage médianes
    for c in prep["num_cols"]:
        if c not in df.columns:
            df[c] = prep["medians"][c]
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(prep["medians"][c])

    return df


# =====================================================================
# --- 3. FONCTIONS D'INFÉRENCE POUR L'API ---
# =====================================================================


def predict_frequency(data_dict: dict) -> float:
    _load_bundles()
    df_input = pd.DataFrame([data_dict])
    processed_df = apply_feature_engineering(df_input)

    # Utilise le préprocesseur spécifique au bundle de fréquence
    X_freq = apply_te_preprocessor(processed_df, _FREQ_BUNDLE["preprocessor"])

    # Suppression des colonnes inutiles
    drop_cols = _FREQ_BUNDLE["features_to_drop"]
    X_freq = X_freq.drop(columns=[c for c in drop_cols if c in X_freq.columns])

    # Prédiction de la probabilité
    prob = _FREQ_BUNDLE["model"].predict_proba(X_freq)[:, 1][0]
    return float(prob)


def predict_severity(data_dict: dict) -> float:
    _load_bundles()
    df_input = pd.DataFrame([data_dict])
    processed_df = apply_feature_engineering(df_input)

    # Utilise le préprocesseur spécifique au bundle de sévérité
    X_sev = apply_te_preprocessor(processed_df, _SEV_BUNDLE["preprocessor"])

    # Suppression des colonnes inutiles
    drop_cols = _SEV_BUNDLE["features_to_drop"]
    X_sev = X_sev.drop(columns=[c for c in drop_cols if c in X_sev.columns])

    # Prédiction (expm1 car entraîné sur log1p)
    mnt_log = _SEV_BUNDLE["model"].predict(X_sev)[0]
    mnt_final = np.expm1(mnt_log)
    return max(0.0, float(mnt_final))


def get_alpha() -> float:
    """Récupère l'hyperparamètre alpha stocké dans le bundle."""
    _load_bundles()
    return _SEV_BUNDLE.get("best_alpha", 1.0)
