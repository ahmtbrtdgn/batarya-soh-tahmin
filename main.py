from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

model = tf.keras.models.load_model('batarya_soh_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SOHRequest(BaseModel):
    soh_history: list[float]

@app.get("/")
def root():
    return {"mesaj": "Batarya SOH API çalışıyor! ✅"}

@app.post("/predict")
def predict_soh(request: SOHRequest):
    data = np.array(request.soh_history).reshape(-1, 1)
    data_normalized = scaler.transform(data).flatten()
    X = data_normalized.reshape(1, 10, 1)
    
    pred_normalized = model.predict(X)
    pred_soh = scaler.inverse_transform(pred_normalized)[0][0]
    
    
    soh_array = np.array(request.soh_history)
    dusus_hizi = (soh_array[0] - soh_array[-1]) / len(soh_array)
    
    if dusus_hizi > 0:
        rul = int((pred_soh - 80) / dusus_hizi)
        rul = max(0, rul)
    else:
        rul = 999

    if pred_soh >= 90:
        durum = "Mükemmel 🟢"
    elif pred_soh >= 80:
        durum = "İyi 🟡"
    elif pred_soh >= 70:
        durum = "Dikkat ⚠️"
    else:
        durum = "Kritik 🔴"

    return {
        "tahmin_soh": round(float(pred_soh), 2),
        "rul": rul,
        "durum": durum,
        "eol_gecildi": bool(pred_soh < 80)
    }