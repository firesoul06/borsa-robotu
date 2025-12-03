import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import time
from datetime import datetime

# TensorFlow loglarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# --- AYARLAR ---
HISSE_KODU = "FROTO.IS"
ATR_CARPANI = 3.0
MODEL_FILE = "froto_beyni.keras"
SCALER_FILE = "froto_scaler.joblib"

st.set_page_config(page_title="Borsa Robotu Pro", page_icon="🤖", layout="centered")

# --- FONKSİYONLAR ---

def veri_getir_ve_isles(period="3mo"): # Eğitim için 3mo, tahmin için 3mo yeterli
    # Yahoo Finance'den veri çek (15 dakika gecikmeli olabilir)
    df = yf.download(HISSE_KODU, period=period, interval="1h", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    if df.empty:
        return None

    # İndikatörler (Robot.py ile AYNI)
    df = df[df['Volume'] > 0]
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)

    df['return_1h'] = df['Close'].pct_change(1)
    df['return_3h'] = df['Close'].pct_change(3)
    df['return_5h'] = df['Close'].pct_change(5)
    df['return_10h'] = df['Close'].pct_change(10)
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    df['volatility_10h'] = df['return_1h'].rolling(10).std()

    # Zaman (UTC'den İstanbul'a)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Europe/Istanbul')

    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df = df.drop('hour', axis=1)
    
    df.dropna(inplace=True)
    return df

def modeli_egit():
    status_text = st.empty()
    bar = st.progress(0)
    
    status_text.text("Veriler indiriliyor...")
    # Eğitim için daha uzun veri çekiyoruz (2 Yıl - 59 gün sınırı 15dk için geçerli, 1h için 730d ok)
    df = veri_getir_ve_isles(period="730d")
    
    if df is None:
        st.error("Veri çekilemedi.")
        return False

    bar.progress(20)
    status_text.text("Veri hazırlanıyor...")

    # Hedef Belirleme
    df['MA_Fast'] = df['Close'].rolling(10).mean()
    df['MA_Slow'] = df['Close'].rolling(30).mean()
    df.dropna(inplace=True)
    
    df['Target'] = 0
    df.loc[df['MA_Fast'] > df['MA_Slow'], 'Target'] = 1
    
    # Sadece Features (Target ve MA hariç)
    X = df.drop(['Target', 'MA_Fast', 'MA_Slow'], axis=1)
    y = df['Target']
    
    # Scale Et
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    dump(scaler, SCALER_FILE) # Kaydet
    
    # LSTM Formatı
    X_lstm, y_lstm = [], []
    TIME_STEPS = 60
    for i in range(len(X_scaled) - TIME_STEPS):
        X_lstm.append(X_scaled[i:(i + TIME_STEPS)])
        y_lstm.append(y.values[i + TIME_STEPS])
    
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    bar.progress(50)
    status_text.text("Yapay Zeka eğitiliyor (Bu işlem sunucuda 1-2 dk sürebilir)...")
    
    # Model Mimarisi
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(60, X.shape[1])))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Hızlı Eğitim (Sunucuyu yormamak için epoch düşük tutulabilir veya EarlyStopping)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_lstm, y_lstm, epochs=15, batch_size=32, verbose=0, callbacks=[early_stop])
    
    model.save(MODEL_FILE)
    
    bar.progress(100)
    status_text.text("Eğitim Tamamlandı! Sayfa yenileniyor...")
    time.sleep(1)
    st.rerun()

# --- ARAYÜZ ---
st.title(f"🤖 {HISSE_KODU} Borsa Robotu")

col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("🔄 Analiz Et (Sinyal Üret)"):
    if not os.path.exists(MODEL_FILE):
        st.warning("Model bulunamadı. Lütfen önce 'Modeli Eğit' butonuna basın.")
    else:
        with st.spinner("Piyasa verileri analiz ediliyor..."):
            df = veri_getir_ve_isles(period="3mo")
            
            if df is not None:
                # Veri ve Model Hazırlığı
                try:
                    scaler = load(SCALER_FILE)
                    model = load_model(MODEL_FILE)
                    
                    last_60_bars = df.iloc[-60:]
                    last_60_scaled = scaler.transform(last_60_bars)
                    input_data = np.array([last_60_scaled])
                    
                    # Tahmin
                    prediction_prob = model.predict(input_data, verbose=0)[0][0]
                    prediction_class = 1 if prediction_prob > 0.5 else 0
                    
                    # Fiyat ve Stop
                    current_price = float(df['Close'].iloc[-1]) # float() garantisi
                    current_atr = float(df['ATRr_14'].iloc[-1])
                    stop_level = current_price - (current_atr * ATR_CARPANI)
                    last_time = df.index[-1]
                    
                    # --- GÖRSELLEŞTİRME ---
                    st.success("Analiz Tamamlandı")
                    
                    # Metrikler
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Anlık Fiyat", f"{current_price:.2f} TL")
                    
                    if prediction_class == 1:
                        m2.metric("Sinyal", "AL 🚀", delta="Yükseliş Beklentisi", delta_color="normal")
                    else:
                        m2.metric("Sinyal", "SAT / BEKLE 🔻", delta="Düşüş Riski", delta_color="inverse")
                        
                    guven = prediction_prob * 100 if prediction_class == 1 else (1-prediction_prob)*100
                    m3.metric("Güven Skoru", f"%{guven:.1f}")
                    
                    st.divider()
                    st.subheader(f"🛡️ Güvenli Stop Noktası: **{stop_level:.2f} TL**")
                    st.caption(f"Veri Zamanı: {last_time.strftime('%d-%m-%Y %H:%M')}")
                    
                    # Grafik (Bonus)
                    st.line_chart(df['Close'].tail(50))
                    
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")
                    st.info("Model dosyaları eski kalmış olabilir. 'Modeli Sıfırdan Eğit' butonunu deneyin.")

if col_btn2.button("🧠 Modeli Sıfırdan Eğit (Reset)"):
    modeli_egit()

st.info("Not: GitHub'daki model dosyaları eskimiş olabilir. En sağlıklı sonuç için sunucuda 'Modeli Sıfırdan Eğit' butonuna basarak güncel verilerle beyni tazeleyin.")