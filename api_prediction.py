from pydantic import BaseModel
from fastapi import FastAPI
from inference import inference
import uvicorn

app = FastAPI()

class Text(BaseModel):
    text: str

@app.post("/predict")
def predict_hate(item:Text):

    # Predict hate speech
    prediction = inference(item.text)

    return prediction
