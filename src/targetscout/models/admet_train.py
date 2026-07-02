"""Train + benchmark an ADMET model on a TDC endpoint, logged to MLflow.

Usage:
    python -m targetscout.models.admet_train --endpoint hERG

Classification endpoints (hERG, DILI, BBB_Martins, ...) report ROC-AUC.
Regression endpoints (Solubility_AqSolDB, Lipophilicity_AstraZeneca, ...) report MAE.
"""
from __future__ import annotations
import argparse
from pathlib import Path

CLASSIFICATION = {"hERG", "DILI", "BBB_Martins", "Bioavailability_Ma", "CYP2D6_Veith",
                  "CYP3A4_Veith", "Pgp_Broccatelli", "AMES"}

MODELS_DIR = Path("artifacts/admet")


def main(endpoint: str) -> None:
    import joblib
    import mlflow
    import numpy as np
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.model_selection import train_test_split

    from targetscout.data.tdc_loader import load_admet
    from targetscout.embeddings.molecule import featurize_many

    is_clf = endpoint in CLASSIFICATION
    train_val, test = load_admet(endpoint)

    Xtr, keep_tr = featurize_many(train_val["Drug"].tolist())
    ytr = train_val["Y"].to_numpy()[keep_tr]
    Xte, keep_te = featurize_many(test["Drug"].tolist())
    yte = test["Y"].to_numpy()[keep_te]

    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.1, random_state=42)

    mlflow.set_experiment("targetscout-admet")
    with mlflow.start_run(run_name=endpoint):
        mlflow.log_params({"endpoint": endpoint, "task": "clf" if is_clf else "reg",
                           "featurizer": "rdkit+morgan1024", "algo": "lightgbm"})
        Model = LGBMClassifier if is_clf else LGBMRegressor
        model = Model(n_estimators=400, learning_rate=0.05, num_leaves=64)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)])

        if is_clf:
            proba = model.predict_proba(Xte)[:, 1]
            metric = roc_auc_score(yte, proba)
            mlflow.log_metric("test_roc_auc", metric)
            print(f"[{endpoint}] test ROC-AUC = {metric:.3f}")
        else:
            pred = model.predict(Xte)
            metric = mean_absolute_error(yte, pred)
            mlflow.log_metric("test_mae", metric)
            print(f"[{endpoint}] test MAE = {metric:.3f}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{endpoint}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(model, "model", registered_model_name=f"admet-{endpoint}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="hERG")
    main(ap.parse_args().endpoint)
