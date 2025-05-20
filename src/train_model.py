import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# === 1. Chargement des données ===
data_path = os.path.join("data", "raw", "donnees_salle.csv")
df = pd.read_csv(data_path)

# === 2. Encodage des variables catégorielles ===
label_encoders = {}
for col in ['sexe', 'objectif', 'entrainement_recommande']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === 3. Séparation des features et de la cible ===
X = df.drop(columns=["entrainement_recommande"])
y = df["entrainement_recommande"]

# === 4. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Entraînement du modèle ===
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# === 6. Évaluation ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# === 7. Sauvegarde du modèle et des encodeurs ===
os.makedirs("data/processed", exist_ok=True)
joblib.dump(model, "data/processed/model.joblib")
joblib.dump(label_encoders, "data/processed/encoders.joblib")
