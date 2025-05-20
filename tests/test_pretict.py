from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "age": 30,
        "sexe": "M",
        "objectif": "perte de poids",
        "poids": 82,
        "taille": 178
    })

    assert response.status_code == 200
    assert "entrainement_recommande" in response.json()
