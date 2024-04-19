from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("/app/model/model.joblib")  # Path to your model

@app.get("/predict/")
def predict(features: str):
    features = np.array([float(f) for f in features.split(",")]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
