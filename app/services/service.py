import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TarificationEngine:
    """
    Moteur de tarification d'assurance automobile.
    Cette classe gère le chargement des modèles en mémoire, l'ingénierie
    des caractéristiques, et les prédictions de fréquence et sévérité.
    """

    def __init__(
        self,
        freq_path: str = "app/data/models/model_frequence.pkl",
        sev_path: str = "app/data/models/model_severite.pkl",
    ):
        """
        Initialise le moteur en chargeant les modèles (bundles) en mémoire.

        Args:
            freq_path (str): Chemin vers le fichier pickle du modèle de fréquence.
            sev_path (str): Chemin vers le fichier pickle du modèle de sévérité.
        """
        self._freq_bundle: Dict[str, Any] = {}
        self._sev_bundle: Dict[str, Any] = {}
        self._load_bundles(freq_path, sev_path)

    def _load_bundles(self, freq_path: str, sev_path: str) -> None:
        """
        Charge physiquement les fichiers pickle depuis le disque.
        Cette méthode est privée (indiquée par le '_') et appelée automatiquement au démarrage.

        """
        logger.info("Chargement des bundles modèles")
        logger.info("Chemin modèle fréquence : %s", freq_path)
        logger.info("Chemin modèle sévérité : %s", sev_path)
        try:
            with open(freq_path, "rb") as f:
                self._freq_bundle = pickle.load(f)
            with open(sev_path, "rb") as f:
                self._sev_bundle = pickle.load(f)
            logger.info("Bundles chargés avec succès")
        except FileNotFoundError as e:
            raise RuntimeError(f"Fichier modèle .pkl introuvable. Erreur: {e}")

    @staticmethod
    def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les transformations métiers (Feature Engineering) sur les données brutes.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données brutes du client.

        Returns:
            pd.DataFrame: Le DataFrame enrichi avec de nouvelles caractéristiques.
        """
        df = df.copy()

        df["permis_par_age"] = df["anciennete_permis1"] / (df["age_conducteur1"] + 1)
        df["log_prix"] = np.log1p(df["prix_vehicule"])
        df["vehicule_puissant"] = (df["din_vehicule"] > 120).astype(int)
        df["vitesse_clip"] = df["vitesse_vehicule"].clip(50, 220)
        df["vitesse_vehicule"] = df["vitesse_vehicule"].clip(10, 250)
        df["exposition"] = (df["duree_contrat"] / 12).clip(lower=0.25)
        logger.info("Feature engineering terminé")
        return df

    @staticmethod
    def apply_te_preprocessor(df: pd.DataFrame, prep: Dict[str, Any]) -> pd.DataFrame:
        """
        Applique le Target Encoding sur les variables catégorielles et remplit les valeurs manquantes.

        Args:
            df (pd.DataFrame): Le DataFrame à encoder.
            prep (Dict[str, Any]): Le dictionnaire de préprocessing contenu dans le bundle.

        Returns:
            pd.DataFrame: Le DataFrame encodé, prêt pour l'inférence.
        """
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

    def predict_frequency(self, data_dict: Dict[str, Any]) -> float:
        """
        Prédit la probabilité qu'un sinistre survienne.
        """
        logger.info("Prédiction de fréquence lancée")
        logger.debug("Données reçues pour fréquence : %s", data_dict)

        try:
            df_input = pd.DataFrame([data_dict])
            processed_df = self.apply_feature_engineering(df_input)

            X_freq = self.apply_te_preprocessor(
                processed_df, self._freq_bundle["preprocessor"]
            )

            drop_cols = self._freq_bundle["features_to_drop"]
            X_freq = X_freq.drop(columns=[c for c in drop_cols if c in X_freq.columns])

            prob = self._freq_bundle["model"].predict_proba(X_freq)[:, 1][0]

            logger.info("Prédiction fréquence réussie : %.6f", prob)
            return float(prob)

        except Exception as e:
            logger.exception("Erreur lors de la prédiction de fréquence")
            raise RuntimeError(f"Erreur prédiction fréquence : {e}")

    def predict_severity(self, data_dict: Dict[str, Any]) -> float:
        """
        Prédit le coût estimé du sinistre s'il survient.
        """
        logger.info("Prédiction de sévérité lancée")
        logger.debug("Données reçues pour sévérité : %s", data_dict)

        try:
            df_input = pd.DataFrame([data_dict])
            processed_df = self.apply_feature_engineering(df_input)

            X_sev = self.apply_te_preprocessor(
                processed_df, self._sev_bundle["preprocessor"]
            )

            drop_cols = self._sev_bundle["features_to_drop"]
            X_sev = X_sev.drop(columns=[c for c in drop_cols if c in X_sev.columns])

            mnt_log = self._sev_bundle["model"].predict(X_sev)[0]
            mnt_final = np.expm1(mnt_log)
            mnt_final = max(0.0, float(mnt_final))

            logger.info("Prédiction sévérité réussie : %.2f", mnt_final)
            return mnt_final

        except Exception as e:
            logger.exception("Erreur lors de la prédiction de sévérité")
            raise RuntimeError(f"Erreur prédiction sévérité : {e}")

    def get_alpha(self) -> float:
        """
        Récupère l'hyperparamètre de calibration alpha depuis le bundle de sévérité.

        Returns:
            float: La valeur d'alpha (par défaut 1.0).
        """
        return float(self._sev_bundle.get("best_alpha", 1.0))
