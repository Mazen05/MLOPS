# src/build_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

df = pd.read_csv("data/raw/donnees_salle.csv")

label_encoders = {}
for col in ['sexe', 'objectif', 'entrainement_recommande']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["entrainement_recommande"])
y = df["entrainement_recommande"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

os.makedirs("data/processed", exist_ok=True)
joblib.dump(model, "data/processed/model.joblib")
joblib.dump(label_encoders, "data/processed/encoders.joblib")
