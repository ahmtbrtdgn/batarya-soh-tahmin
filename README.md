# 🔋 Batarya SOH Tahmin Sistemi

NASA Li-ion batarya verisi üzerinde LSTM derin öğrenme modeli ile batarya sağlık durumu (SOH) ve kalan kullanılabilir ömür (RUL) tahmini yapan sistem.

## 📊 Proje Hakkında

Bu proje, NASA Prognostics Center of Excellence'ın Li-ion batarya veri setini kullanarak:
- Batarya **State of Health (SOH)** tahminini
- **Remaining Useful Life (RUL)** tahminini
- Gerçek zamanlı web arayüzü ile kullanıcıya sunmayı

amaçlamaktadır.

## 🎯 Sonuçlar

| Metrik | Değer |
|---|---|
| MAE | %1.03 |
| RMSE | %1.49 |
| Eğitim verisi | 4 batarya (B0005, B0006, B0007, B0018) |
| Toplam döngü | 636 discharge döngüsü |

## 🛠️ Kullanılan Teknolojiler

- **Python** — Ana dil
- **TensorFlow/Keras** — LSTM modeli
- **FastAPI** — REST API backend
- **scikit-learn** — Veri ön işleme
- **Matplotlib** — Veri görselleştirme
- **HTML/CSS/JS** — Web arayüzü

## 📁 Veri Seti

[NASA Battery Dataset](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip) — NASA Prognostics Center of Excellence

4 farklı 18650 Li-ion batarya üzerinde şarj/deşarj deneyleri içermektedir.

## 🚀 Kurulum

### Gereksinimler
```bash
pip install tensorflow fastapi uvicorn scikit-learn scipy numpy matplotlib
```

### Modeli Eğit
```bash
jupyter notebook batarya_soh.ipynb
```

### API'yi Başlat
```bash
uvicorn main:app --reload
```

### Arayüzü Aç
`index.html` dosyasını tarayıcıda aç.

## 📡 API Kullanımı

```bash
POST /predict
{
  "soh_history": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
}
```

**Yanıt:**
```json
{
  "tahmin_soh": 92.83,
  "rul": 45,
  "durum": "İyi 🟡",
  "eol_gecildi": false
}
```

## 🗺️ Yol Haritası

- [x] NASA veri seti analizi
- [x] LSTM modeli eğitimi
- [x] FastAPI backend
- [x] Web arayüzü
- [ ] STM32F4 donanım entegrasyonu
- [ ] Gerçek zamanlı ölçüm
- [ ] Mobil uygulama

## 👨‍💻 Geliştirici

EEE 3. sınıf öğrencisi tarafından geliştirilmiştir.