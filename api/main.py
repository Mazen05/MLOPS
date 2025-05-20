from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = None
encoders = None

@app.post("/predict")
def predict_entrainement(input: dict):
    global model, encoders

    if model is None or encoders is None:
        model = joblib.load("data/processed/model.joblib")
        encoders = joblib.load("data/processed/encoders.joblib")

    input_df = pd.DataFrame([input])
    for col in ['sexe', 'objectif']:
        input_df[col] = encoders[col].transform(input_df[col])

    pred = model.predict(input_df)[0]
    label = encoders['entrainement_recommande'].inverse_transform([pred])[0]
    return {"entrainement_recommande": label}
