import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from joblib import load
import os
from datetime import datetime

# --- AYARLAR ---
HISSE_KODU = "FROTO.IS"
ATR_CARPANI = 3.0
BASE_NAME = HISSE_KODU.replace(".IS", "")
MODEL_FILE = f"{BASE_NAME}_beyni.keras"
SCALER_FILE = f"{BASE_NAME}_scaler.joblib"

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Borsa Robotu AI", page_icon="📈", layout="centered")

st.title(f"🤖 Yapay Zeka Borsa Robotu")
st.subheader(f"Takip Edilen: {HISSE_KODU}")

# --- MODEL YÜKLEME (CACHE İLE HIZLANDIRMA) ---
@st.cache_resource
def dosyalari_yukle():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None
    model = load_model(MODEL_FILE)
    scaler = load(SCALER_FILE)
    return model, scaler

model, scaler = dosyalari_yukle()

if model is None:
    st.error("HATA: Model dosyaları (.keras / .joblib) bulunamadı! Lütfen GitHub'a yüklediğinden emin ol.")
    st.stop()

# --- ANALİZ BUTONU ---
if st.button("🔄 Analizi Başlat / Yenile"):
    with st.spinner('Piyasa verileri çekiliyor ve Yapay Zeka düşünüyor...'):
        # 1. VERİ İNDİRME
        df = yf.download(HISSE_KODU, period="3mo", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # 2. ÖZELLİK MÜHENDİSLİĞİ
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

        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Europe/Istanbul')
        
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df = df.drop('hour', axis=1)
        df.dropna(inplace=True)

        # 3. TAHMİN
        last_60_bars = df.iloc[-60:]
        last_60_scaled = scaler.transform(last_60_bars)
        input_data = np.array([last_60_scaled])
        
        prediction_prob = model.predict(input_data, verbose=0)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0

        # 4. STOP SEVİYESİ
        current_price = df['Close'].iloc[-1]
        current_atr = df['ATRr_14'].iloc[-1]
        stop_level = current_price - (current_atr * ATR_CARPANI)
        
        # --- GÖRSELLEŞTİRME ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anlık Fiyat", f"{current_price:.2f} TL")
        
        with col2:
            if prediction_class == 1:
                st.metric("AI Tahmini", "YÜKSELİŞ 🚀", delta_color="normal")
            else:
                st.metric("AI Tahmini", "DÜŞÜŞ 🔻", delta_color="inverse")
                
        with col3:
            guven = prediction_prob * 100 if prediction_class == 1 else (1-prediction_prob) * 100
            st.metric("Güven Oranı", f"%{guven:.1f}")

        st.divider()
        
        # Tavsiye Kutusu
        if prediction_class == 1:
            st.success(f"💡 **TAVSİYE:** Model pozitif. Alım düşünülebilir.\n\n🛡️ **Stop Loss:** {stop_level:.2f} TL")
        else:
            st.error(f"💡 **TAVSİYE:** Model negatif. Nakitte kal veya sat.\n\n🛡️ **Stop Loss:** {stop_level:.2f} TL")

        st.divider()
        st.caption(f"Son Veri Zamanı: {df.index[-1]}")