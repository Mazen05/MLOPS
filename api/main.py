from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Chargement du modèle et des encodeurs
model = None
encoders = joblib.load("data/processed/encoders.joblib")

# Schéma d’entrée de l’utilisateur
class UserInput(BaseModel):
    age: int
    sexe: str
    objectif: str
    poids: float
    taille: int

@app.post("/predict")
def predict_entrainement(user: UserInput):
    # Encodage des variables
    global model
if model is None:
    model = joblib.load("data/processed/model.joblib")

    sexe_encoded = encoders["sexe"].transform([user.sexe])[0]
    objectif_encoded = encoders["objectif"].transform([user.objectif])[0]

    input_data = np.array([[user.age, sexe_encoded, objectif_encoded, user.poids, user.taille]])

    # Prédiction
    pred = model.predict(input_data)[0]
    entrainement = encoders["entrainement_recommande"].inverse_transform([pred])[0]

    return {"entrainement_recommande": entrainement}
