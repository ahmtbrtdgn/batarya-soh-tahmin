import gradio as gr
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import urllib.request
import zipfile
import os
import glob

def train_and_save():
    if not os.path.exists('battery.zip'):
        print("Dataset indiriliyor...")
        urllib.request.urlretrieve(
            "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
            "battery.zip"
        )
    
    if not os.path.exists('B0005.mat'):
        print("Zip açılıyor...")
        with zipfile.ZipFile("battery.zip", 'r') as z:
            # Zip içeriğini listele
            all_files = z.namelist()
            print("Zip içindekiler:", all_files[:20])
            z.extractall(".")
        
        # Tüm dosyaları listele
        import shutil
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith('.mat'):
                    full_path = os.path.join(root, f)
                    print(f"MAT bulundu: {full_path}")
                    shutil.copy(full_path, f)

    def get_soh(mat_file, key):
        mat = scipy.io.loadmat(mat_file)
        cycles = mat[key][0][0]['cycle'][0]
        capacities = []
        for cycle in cycles:
            if cycle['type'][0] == 'discharge':
                data = cycle['data'][0][0]
                current = data['Current_measured'][0].flatten()
                time = data['Time'][0].flatten()
                capacity = abs(np.trapezoid(current, time)) / 3600
                capacities.append(capacity)
        capacities = np.array(capacities)
        return (capacities / capacities[0]) * 100

    files = {
        'B0005': 'B0005.mat',
        'B0006': 'B0006.mat',
        'B0007': 'B0007.mat',
        'B0018': 'B0018.mat',
    }
    
    all_soh = np.concatenate([get_soh(v, k) for k, v in files.items()])

    scaler = MinMaxScaler()
    all_soh_norm = scaler.fit_transform(all_soh.reshape(-1, 1)).flatten()

    SEQ_LENGTH = 10
    X, y = [], []
    for i in range(len(all_soh_norm) - SEQ_LENGTH):
        X.append(all_soh_norm[i:i+SEQ_LENGTH])
        y.append(all_soh_norm[i+SEQ_LENGTH])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, SEQ_LENGTH, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    model.save('batarya_soh_model.keras')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

if os.path.exists('batarya_soh_model.keras'):
    model = tf.keras.models.load_model('batarya_soh_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
else:
    print("Model eğitiliyor...")
    model, scaler = train_and_save()

def predict_soh(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10):
    soh_history = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
    data = np.array(soh_history).reshape(-1, 1)
    data_normalized = scaler.transform(data).flatten()
    X = data_normalized.reshape(1, 10, 1)
    pred_normalized = model.predict(X)
    pred_soh = scaler.inverse_transform(pred_normalized)[0][0]

    soh_array = np.array(soh_history)
    dusus_hizi = (soh_array[0] - soh_array[-1]) / len(soh_array)
    rul = int((pred_soh - 80) / dusus_hizi) if dusus_hizi > 0 else 999
    rul = max(0, rul)

    if pred_soh >= 90:
        durum = "Mükemmel 🟢"
    elif pred_soh >= 80:
        durum = "İyi 🟡"
    elif pred_soh >= 70:
        durum = "Dikkat ⚠️"
    else:
        durum = "Kritik 🔴"

    eol = "⚠️ EOL Sınırı Aşıldı!" if pred_soh < 80 else "✅ EOL Sınırı İçinde"
    return round(float(pred_soh), 2), rul, durum, eol

with gr.Blocks(title="Batarya SOH Tahmin Sistemi") as demo:
    gr.Markdown("# 🔋 Batarya SOH Tahmin Sistemi")
    gr.Markdown("Son 10 döngünün SOH değerlerini girin, LSTM modeli tahmin yapsın.")
    with gr.Row():
        d1 = gr.Number(label="Döngü 1", value=100)
        d2 = gr.Number(label="Döngü 2", value=99)
        d3 = gr.Number(label="Döngü 3", value=98)
        d4 = gr.Number(label="Döngü 4", value=97)
        d5 = gr.Number(label="Döngü 5", value=96)
    with gr.Row():
        d6 = gr.Number(label="Döngü 6", value=95)
        d7 = gr.Number(label="Döngü 7", value=94)
        d8 = gr.Number(label="Döngü 8", value=93)
        d9 = gr.Number(label="Döngü 9", value=92)
        d10 = gr.Number(label="Döngü 10", value=91)
    btn = gr.Button("🔍 Tahmin Et", variant="primary")
    with gr.Row():
        out_soh = gr.Number(label="Tahmini SOH (%)")
        out_rul = gr.Number(label="Kalan Ömür (döngü)")
        out_durum = gr.Text(label="Durum")
        out_eol = gr.Text(label="EOL Durumu")
    btn.click(fn=predict_soh, inputs=[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],
              outputs=[out_soh, out_rul, out_durum, out_eol])

demo.launch()