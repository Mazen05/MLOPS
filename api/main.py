from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Exemple de schéma d'entrée.
class InputData(BaseModel):
    age: int
    sexe: str
    objectif: str
    poids: float
    taille: float

# Load le modèle une fois au démarrage
try:
    model = joblib.load("data/processed/model.joblib")
    label_encoder = joblib.load("data/processed/label_encoder.joblib")
except Exception as e:
    model = None
    label_encoder = None
    print(f"Erreur de chargement du modèle: {e}")

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps 🎯"}

@app.get("/health")
def health_check():
    if model:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False}

@app.post("/predict")
def predict_entrainement(data: InputData): #fff
    if not model:
        return {"error": "Modèle non chargé."}

    sexe = 1 if data.sexe.lower() == "m" else 0
    objectif_enc = label_encoder.transform([data.objectif])[0]

    features = np.array([[data.age, sexe, objectif_enc, data.poids, data.taille]])
    pred = model.predict(features)[0]
    return {"entrainement_recommande": pred}
