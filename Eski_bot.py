import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf
import random
import os
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
from datetime import datetime

# Uyarıları kapat
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Sayfa Konfigürasyonu
st.set_page_config(
    page_title="AI Borsa Robotu",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sabitler ve Ayarlar
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# --- YARDIMCI FONKSİYONLAR ---

def veri_getir_ve_isleo(hisse_kodu, sure, aralik):
    """Veriyi indirir ve indikatörleri hesaplar."""
    df = yf.download(hisse_kodu, period=sure, interval=aralik, progress=False)
    
    # MultiIndex düzeltmesi
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Hacim 0 temizliği
    df = df[df['Volume'] > 0]
    
    if df.empty:
        return None

    # Teknik İndikatörler
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)

    # Momentum & Volatilite
    df['return_1h'] = df['Close'].pct_change(1)
    df['return_3h'] = df['Close'].pct_change(3)
    df['return_5h'] = df['Close'].pct_change(5)
    df['return_10h'] = df['Close'].pct_change(10)
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    df['volatility_10h'] = df['return_1h'].rolling(10).std()

    # Zaman Döngüsü
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Europe/Istanbul')

    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df = df.drop('hour', axis=1)
    
    # NaN temizliği
    df.dropna(inplace=True)
    return df

def create_dataset(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- MODÜL 1: EĞİTİM ---
def egitim_modulu():
    st.subheader("🛠️ Model Eğitim Fabrikası")
    st.info("Bu modül, seçilen hisse senedi için geçmiş verileri kullanarak yeni bir yapay zeka modeli eğitir.")

    col1, col2 = st.columns(2)
    with col1:
        hisse_kodu = st.text_input("Hisse Kodu (Örn: FROTO.IS)", "FROTO.IS")
    with col2:
        veri_suresi = st.selectbox("Veri Geçmişi", ["59d", "1y", "2y"], index=0)

    if st.button("Eğitimi Başlat", type="primary"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.text("Veriler indiriliyor...")
        
        # Dosya İsimleri
        base_name = hisse_kodu.replace(".IS", "")
        model_file = f"{base_name}_beyni.keras"
        scaler_file = f"{base_name}_scaler.joblib"

        try:
            # Adım 1: Veri Toplama
            df = veri_getir_ve_isleo(hisse_kodu, veri_suresi, "15m")
            
            if df is None or len(df) < 200:
                st.error("Yetersiz veri veya hatalı hisse kodu.")
                return

            status_text.text(f"Veri işleniyor... ({len(df)} satır)")
            progress_bar.progress(25)

            # Adım 2: Hedef Belirleme (Target)
            ma_fast = 10
            ma_slow = 30
            df['MA_Fast'] = df['Close'].rolling(ma_fast).mean()
            df['MA_Slow'] = df['Close'].rolling(ma_slow).mean()
            df.dropna(inplace=True)

            df['Target'] = 0
            df.loc[df['MA_Fast'] > df['MA_Slow'], 'Target'] = 1
            
            # Kopya çekmeyi engelle
            df_train = df.drop(['MA_Fast', 'MA_Slow'], axis=1)

            # Adım 3: Scaling & Split
            X = df_train.drop('Target', axis=1)
            y = df_train['Target']

            train_size = int(len(X) * 0.8)
            X_train_raw = X.iloc[:train_size]
            X_test_raw = X.iloc[train_size:]
            y_train_raw = y.iloc[:train_size].values
            y_test_raw = y.iloc[train_size:].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            dump(scaler, scaler_file)
            status_text.text("Scaler kaydedildi, veri seti oluşturuluyor...")
            progress_bar.progress(50)

            # LSTM Hazırlık
            time_steps = 60
            X_train_lstm, y_train_lstm = create_dataset(X_train_scaled, y_train_raw, time_steps)
            X_test_lstm, y_test_lstm = create_dataset(X_test_scaled, y_test_raw, time_steps)

            # Adım 4: Model Mimarisi
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(100, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=True, verbose=0)

            status_text.text("Model eğitiliyor (Bu işlem biraz sürebilir)...")
            
            # Streamlit üzerinde eğitimi göstermek zor olduğu için verbose=0 yapıp sonucu bekliyoruz
            with st.spinner('Yapay zeka piyasa hareketlerini öğreniyor...'):
                history = model.fit(
                    X_train_lstm, y_train_lstm,
                    epochs=50, # Hız için 50'ye düşürdüm, istersen arttır
                    batch_size=32,
                    validation_data=(X_test_lstm, y_test_lstm),
                    verbose=0,
                    callbacks=[early_stopping, model_checkpoint]
                )

            progress_bar.progress(100)
            status_text.text("Tamamlandı!")
            
            st.success(f"✅ Eğitim Başarılı! Model: {model_file}")
            
            # Eğitim Başarısı Grafiği
            loss_df = pd.DataFrame(history.history)
            st.line_chart(loss_df[['loss', 'val_loss']])
            st.caption("Eğitim Kayıp Grafiği (Düşük olması iyidir)")

        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")

# --- MODÜL 2: ROBOT ---
def robot_modulu():
    st.subheader("🤖 Analiz Robotu")
    st.info("Eğitilmiş modeli kullanarak anlık analiz yapar ve Al/Sat sinyali üretir.")

    hisse_kodu = st.text_input("Analiz Edilecek Hisse", "FROTO.IS")
    atr_carpani = st.slider("ATR Stop Çarpanı", 1.0, 5.0, 3.0)

    base_name = hisse_kodu.replace(".IS", "")
    model_file = f"{base_name}_beyni.keras"
    scaler_file = f"{base_name}_scaler.joblib"

    # Dosya Kontrolü
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        st.warning(f"⚠️ {hisse_kodu} için eğitilmiş model bulunamadı! Önce 'Eğitim Modu'na giderek modeli eğitin.")
        return

    if st.button("Analiz Et", type="primary"):
        with st.spinner('Piyasa verileri çekiliyor ve analiz ediliyor...'):
            try:
                # 1. Veri İndirme (Analiz için son 5 gün yeterli)
                df = veri_getir_ve_isleo(hisse_kodu, "5d", "15m")
                
                if df is None:
                    st.error("Veri çekilemedi.")
                    return

                # 2. Tahmin
                last_60_bars = df.iloc[-60:]
                
                # Scaler ve Model Yükleme
                model = load_model(model_file)
                scaler = load(scaler_file)
                
                # Sadece input featureları seç
                feature_columns = [col for col in df.columns if col not in ['Target', 'MA_Fast', 'MA_Slow']]
                # Veri setini eğitirken kullanılan sütun sayısını kontrol etmek gerekebilir, 
                # ancak veri_getir_ve_isleo fonksiyonu standart olduğu için uyumlu olmalı.
                
                last_60_scaled = scaler.transform(last_60_bars[feature_columns]) # Sadece feature sütunları
                
                input_data = np.array([last_60_scaled])
                prediction_prob = model.predict(input_data, verbose=0)[0][0]
                prediction_class = 1 if prediction_prob > 0.5 else 0

                # 3. Raporlama
                current_price = df['Close'].iloc[-1]
                current_atr = df['ATRr_14'].iloc[-1]
                stop_level = current_price - (current_atr * atr_carpani)

                # Görselleştirme
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Anlık Fiyat", f"{current_price:.2f} TL")
                
                with col2:
                    delta_color = "normal"
                    if prediction_class == 1:
                        durum = "YÜKSELİŞ (AL)"
                        delta_color = "off" # yeşilmsi için custom gerekebilir ama normal kalsın
                        st.success(f"Tahmin: {durum}")
                    else:
                        durum = "DÜŞÜŞ (BEKLE)"
                        st.error(f"Tahmin: {durum}")

                with col3:
                    st.metric("Güven Oranı", f"%{prediction_prob*100:.2f}")

                st.divider()
                st.write(f"🛑 **Önerilen Stop Seviyesi:** {stop_level:.2f} TL")
                
                # Son 60 bar grafiği
                st.subheader("Son Fiyat Hareketleri")
                st.line_chart(last_60_bars['Close'])

            except Exception as e:
                st.error(f"Analiz sırasında hata: {str(e)}")
                # Hata ayıklama için detay:
                st.write("Olası neden: Modelin eğitildiği veri yapısı ile şu anki veri yapısı uyuşmuyor olabilir.")

# --- ANA UYGULAMA ---
def main():
    with st.sidebar:
        st.header("Kontrol Paneli")
        secim = st.radio("Mod Seçimi", ["Eğitim Modu", "Robot Modu"])
        st.markdown("---")
        st.caption("Yasal Uyarı: Buradaki veriler yatırım tavsiyesi değildir.")

    st.title("🚀 Borsa Yapay Zeka Asistanı")

    if secim == "Eğitim Modu":
        egitim_modulu()
        
    elif secim == "Robot Modu":
        robot_modulu()

if __name__ == "__main__":
    main()
