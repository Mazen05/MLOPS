from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    age: int
    sexe: str
    objectif: str
    poids: float
    taille: float

# Chargement du modèle
try:
    model = joblib.load("data/processed/model.joblib")
except:
    model = None

# Chargement des encodeurs
try:
    encoders = joblib.load("data/processed/encoders.joblib")
    label_encoder = encoders["objectif"]
except:
    label_encoder = None

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps 🎯"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_entrainement(data: InputData):
    if model is None or label_encoder is None:
        return {"error": "Modèle non chargé."}

    sexe = 1 if data.sexe.lower() == "m" else 0
    objectif_enc = label_encoder.transform([data.objectif])[0]

    features = np.array([[data.age, sexe, objectif_enc, data.poids, data.taille]])
    pred = model.predict(features)[0]

    return {"entrainement_recommande": pred}
