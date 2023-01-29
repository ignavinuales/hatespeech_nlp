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


if __name__ == "__main__":
    uvicorn.run("api_prediction:app", host="0.0.0.0", port=8000, log_level="info")