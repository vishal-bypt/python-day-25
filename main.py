from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()
MODEL_PATH = "sentiment_model.pkl"

# ---------------------------
# Create dummy model if missing
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("⚠️ sentiment_model.pkl not found, creating dummy model...")

    texts = ["I love it", "I hate it", "This is great", "This is terrible"]
    labels = ["positive", "negative", "positive", "negative"]

    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)

    # Save vectorizer + model together
    joblib.dump((vec, model), MODEL_PATH)
    print("✅ Dummy sentiment_model.pkl created!")

# ---------------------------
# Load model
# ---------------------------
vec, model = joblib.load(MODEL_PATH)

# ---------------------------
# Input schema
# ---------------------------
class InputText(BaseModel):
    text: str

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def health_check():
    return {"Health Status": "OK"}

@app.post("/predict")
def predict_sentiment(data: InputText):
    if model is None or vec is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    prediction = model.predict(vec.transform([data.text]))[0]
    return {"sentiment": prediction}
