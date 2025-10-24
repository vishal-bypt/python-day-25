from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("sentiment_model.pkl")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: InputText):
    prediction = model.predict([data.text])[0]
    return {"sentiment": prediction}
