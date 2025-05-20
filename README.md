
# 🏋️ Coach Sportif Intelligent

Ce projet est une application de recommandation d'entraînement basée sur le Machine Learning.

## 📌 Objectif

Recommander automatiquement un type d'entraînement personnalisé en fonction de l'âge, du sexe, du poids, de la taille et de l'objectif de remise en forme.

## 🚀 Lancement rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Génération des données (optionnel)

```bash
python src/generate_data.py
```

### 3. Entraînement du modèle

```bash
python src/train_model.py
```

### 4. Lancer l'API FastAPI

```bash
uvicorn api.main:app --reload
```

### 5. Accéder à la documentation

[http://localhost:8000/docs](http://localhost:8000/docs)

## 🔎 Exemple de requête

```json
{
  "age": 28,
  "sexe": "M",
  "objectif": "perte de poids",
  "poids": 85,
  "taille": 180
}
```

## ✅ Exemple de réponse

```json
{
  "entrainement_recommande": "Circuit Training"
}
```

## 🧪 Tests

```bash
pytest tests/test_predict.py
```
