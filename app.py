import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import datetime

# --- SAYFA AYARLARI (En başta olmalı) ---
st.set_page_config(page_title="ProQuant AI Bot", layout="wide", page_icon="⚡")

# --- CSS STİL ---
st.markdown("""
<style>
    .stMetric { background-color: #0e1117; border: 1px solid #30333F; padding: 15px; border-radius: 10px; }
    .css-1y4p8pa { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ ProQuant AI: Profesyonel Algoritmik Analiz")
st.markdown("Derin Öğrenme (LSTM) | Risk Yönetimi (ATR) | Teknik Analiz")

# --- YAN MENÜ ---
st.sidebar.header("⚙️ Kontrol Paneli")
hisse_kodu = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS").upper()
egitim_yili = st.sidebar.selectbox("Geçmiş Veri Analizi", ["3 Yıl", "5 Yıl"], index=0)
epoch_sayisi = st.sidebar.slider("Eğitim Tekrarı (Epoch)", 20, 60, 30)
analiz_baslat = st.sidebar.button("ANALİZİ BAŞLAT 🚀")

# --- YARDIMCI FONKSİYONLAR ---
def veri_temizle_ve_indir(sembol, yil_secimi):
    yil_map = {"3 Yıl": 3, "5 Yıl": 5}
    start_date = datetime.datetime.now() - datetime.timedelta(days=365*yil_map[yil_secimi])
    
    # Veri indirme
    df = yf.download(sembol, start=start_date, progress=False)
    
    if df.empty:
        return None
        
    # KRİTİK DÜZELTME: MultiIndex Sütunları Düzleştirme
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 'Close' sütunu kontrolü
    if 'Close' not in df.columns:
        return None
        
    return df

def teknik_indikatorler(df):
    # Veri bütünlüğünü korumak için kopya alıyoruz
    data = df.copy()
    
    # 1. RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. ATR (Average True Range) - Volatilite ve Stop Loss için
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    
    # 3. Hareketli Ortalamalar
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # NaN değerleri temizle
    data.dropna(inplace=True)
    return data

def create_sequences(data, look_back=60):
    X, y = [], []
    # Çok değişkenli girdi, tek çıktı (Fiyat)
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i]) 
        y.append(data[i, 0]) # 0. indeks her zaman 'Close' fiyatı olacak şekilde ayarlayacağız
    return np.array(X), np.array(y)

# --- ANA PROGRAM AKIŞI ---
if analiz_baslat:
    try:
        with st.spinner('Veriler borsadan çekiliyor ve işleniyor...'):
            raw_df = veri_temizle_ve_indir(hisse_kodu, egitim_yili)
            
            if raw_df is None:
                st.error("HATA: Veri çekilemedi veya hisse kodu hatalı. Sonuna .IS eklemeyi unutmayın (Örn: GARAN.IS).")
                st.stop()
                
            df = teknik_indikatorler(raw_df)
            
            # Model için kullanılacak özellikler (Sıralama Önemli!)
            # İlk sıraya 'Close' koyuyoruz ki scaler geri dönüşümünde kolay olsun.
            features = ['Close', 'RSI', 'ATR', 'SMA_50', 'Volume']
            
            # Sütunların varlığını kontrol et
            if not all(col in df.columns for col in features):
                st.error("Veri setinde gerekli teknik sütunlar eksik.")
                st.stop()

            dataset = df[features].values
            
            # Ölçeklendirme (Scaling)
            scaler_all = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler_all.fit_transform(dataset)
            
            # Sadece Fiyat için ayrı scaler (Geri dönüşüm için)
            scaler_price = MinMaxScaler(feature_range=(0, 1))
            scaler_price.fit(dataset[:, 0].reshape(-1, 1))
            
            # Eğitim verisi hazırlama
            LOOK_BACK = 60
            X, y = create_sequences(scaled_data, LOOK_BACK)
            
            # Train/Test Split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Şekil düzeltme (LSTM 3D ister: Samples, TimeSteps, Features)
            # X zaten doğru boyutta geliyor ama emin olmak için kontrol edebiliriz
            
    except Exception as e:
        st.error(f"Veri hazırlama aşamasında hata: {e}")
        st.stop()

    # --- MODEL EĞİTİMİ ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Yapay Zeka (Bi-LSTM) Eğitiliyor...")
        
        model = Sequential()
        # Bidirectional LSTM: Geçmişi ve "geleceği" (eğitim setindeki) çift yönlü okur
        model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=32))
        model.add(Dense(units=1)) # Çıkış katmanı (Fiyat)
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early Stopping: Ezberlemeyi önle
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, batch_size=32, epochs=epoch_sayisi, verbose=0, callbacks=[early_stop])
        progress_bar.progress(100)
        status_text.text("Analiz ve Simülasyon Tamamlandı!")
        
    except Exception as e:
        st.error(f"Model eğitimi sırasında hata: {e}")
        st.stop()

    # --- TAHMİN VE SONUÇLAR ---
    try:
        # 1. Backtest Tahminleri (Grafik için)
        predictions = model.predict(X_test)
        predictions_inv = scaler_price.inverse_transform(predictions)
        y_test_inv = scaler_price.inverse_transform(y_test.reshape(-1, 1))
        
        # 2. Gelecek Tahmini (Yarın için)
        last_sequence = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, len(features))
        future_pred_scaled = model.predict(last_sequence)
        future_price = float(scaler_price.inverse_transform(future_pred_scaled)[0][0]) # .item() mantığı
        
        # Güncel değerler (Güvenli çekim)
        current_price = float(df['Close'].iloc[-1].item())
        current_rsi = float(df['RSI'].iloc[-1].item())
        current_atr = float(df['ATR'].iloc[-1].item())
        
        degisim_yuzde = ((future_price - current_price) / current_price) * 100
        
        # --- MANTIK VE GÖRSELLEŞTİRME ---
        st.divider()
        
        # Yön Kararı ve Renk Ayarı (Logic Fix)
        if future_price > current_price:
            trend_yonu = "YUKARI 🟢"
            oneri = "LONG (ALIM)"
            # Stop Loss: Fiyatın altına koyulur
            stop_loss = current_price - (current_atr * 1.5)
            delta_color_val = "normal" # Yeşil pozitif
        else:
            trend_yonu = "AŞAĞI 🔴"
            oneri = "SHORT (SATIŞ/BEKLE)"
            # Stop Loss: Fiyatın üstüne koyulur (Short işlem için)
            stop_loss = current_price + (current_atr * 1.5) 
            delta_color_val = "inverse" # Kırmızı negatif (ama short için yeşil algılatılabilir, biz kırmızıyı tercih edelim uyarı için)

        # KARTLAR
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Anlık Fiyat", f"{current_price:.2f} TL")
            
        with col2:
            st.metric("AI Hedef (1 Ay)", f"{future_price:.2f} TL", f"%{degisim_yuzde:.2f}", delta_color=delta_color_val)
            
        with col3:
            st.metric("Stop-Loss (Risk)", f"{stop_loss:.2f} TL", help="Bu seviye risk yönetimi sınırıdır.")
            
        with col4:
            rsi_durum = "Aşırı Alım 🔴" if current_rsi > 70 else "Aşırı Satım 🟢" if current_rsi < 30 else "Nötr ⚪"
            st.metric("RSI İndikatörü", f"{current_rsi:.1f}", rsi_durum)

        # STRATEJİ RAPORU
        st.subheader(f"📢 Yapay Zeka Stratejisi: {oneri}")
        
        if future_price > current_price:
            st.success(f"Model yükseliş öngörüyor. Fiyatın **{future_price:.2f} TL** seviyesine gitmesi bekleniyor. Risk yönetimi için **{stop_loss:.2f} TL** seviyesine stop-loss konulabilir.")
        else:
            st.error(f"Model düşüş veya düzeltme öngörüyor. Fiyat **{future_price:.2f} TL** seviyelerine gevşeyebilir. Alım için acele etme veya Short pozisyon değerlendir.")

        # GRAFİK (Interactive Plotly)
        st.subheader("🧪 Backtest: Model vs Gerçek Piyasa")
        
        # Tarih dizisi oluştur (Test verisi için)
        dates = df.index[train_size+LOOK_BACK:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_test_inv.flatten(), mode='lines', name='Gerçek Fiyat', line=dict(color='#00CC96', width=2)))
        fig.add_trace(go.Scatter(x=dates, y=predictions_inv.flatten(), mode='lines', name='AI Tahmini', line=dict(color='#EF553B', width=2, dash='dot')))
        
        fig.update_layout(
            title=f"{hisse_kodu} Model Başarısı",
            xaxis_title="Tarih",
            yaxis_title="Fiyat (TL)",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("⚠️ YASAL UYARI: Bu yazılım bir mühendislik projesidir ve eğitim amaçlıdır. Yatırım tavsiyesi değildir.")
        
    except Exception as e:
        st.error(f"Sonuçları gösterirken hata oluştu: {e}")
