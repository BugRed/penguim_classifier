from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from .model_util import load_model, predict_instance

app = FastAPI(title="Pinguim Predictor API")

# Permissões do Streamlit (rodando em outra porta) para chamar API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Não Restrinje os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PenguimInput(BaseModel):
    culmen_length_mm: float = Field(..., gt=0)
    culmen_depth_mm: float = Field(..., gt=0)
    flipper_length_mm: float = Field(..., gt=0)
    body_mass_g: float = Field(..., gt=0)

# Output schema
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float  # 0..1
    probabilities: List[float]  


# Load model on startup
MODEL_PATH = "model/penguim_classifier_tree_model.pkl"
model = load_model(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Pinguim Predictor API — use POST /predict"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PenguimInput):
    x = [
        data.culmen_length_mm,
        data.culmen_depth_mm,
        data.flipper_length_mm,
        data.body_mass_g,
    ]
    pred_class, confidence, probs = predict_instance(model, x)
    return PredictionResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 4),
        probabilities=[round(float(p), 4) for p in probs.tolist()]
    )
