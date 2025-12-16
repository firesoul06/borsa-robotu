import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
import datetime

# --- 1. AYARLAR VE KONFİGÜRASYON ---
st.set_page_config(page_title="ProQuant Terminal", layout="wide", page_icon="⚡")

# Profesyonel Dark Tema CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #e6edf3; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# --- 2. GÜÇLENDİRİLMİŞ VERİ MOTORU ---
@st.cache_data(ttl=60) # 1 Dakika Önbellek (Hız için)
def get_clean_data(symbol, period="2y"):
    try:
        # Veri İndirme
        df = yf.download(symbol, period=period, progress=False)
        
        # HATA KORUMASI 1: MultiIndex Sütun Düzeltme
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # HATA KORUMASI 2: Eksik Veri Kontrolü
        if df.empty or 'Close' not in df.columns:
            return None
            
        # Veri Tiplerini Garantiye Alma
        df = df.astype(float)
            
        # İndikatör Hesaplamaları
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Risk Yönetimi İçin)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

def get_basic_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except:
        return {}

# --- 3. YAN MENÜ ---
st.sidebar.title("⚡ ProQuant AI")
st.sidebar.markdown("---")
ticker = st.sidebar.text_input("Hisse Kodu Girin", value="THYAO.IS").upper()
period = st.sidebar.selectbox("Analiz Aralığı", ["1y", "2y", "5y"], index=1)
btn_analiz = st.sidebar.button("PİYASAYI TARA 🚀")

st.sidebar.info("💡 **İpucu:** BIST hisseleri için sonuna .IS ekleyin (Örn: `ASELS.IS`, `ALTIN.IS`). Kripto için `BTC-USD`.")

# --- 4. ANA PROGRAM ---
if btn_analiz or ticker:
    
    # Veri Yükleniyor Animasyonu
    with st.spinner(f"'{ticker}' için yapay zeka ve piyasa verileri işleniyor..."):
        df = get_clean_data(ticker, period)
        info = get_basic_info(ticker)
    
    # Eğer veri yoksa veya hata varsa
    if df is None or len(df) < 60:
        st.error("⛔ Veri bulunamadı veya analiz için yeterli tarihçe yok (En az 60 gün gerekli). Hisse kodunu kontrol edin.")
        st.stop()

    # --- ÜST BİLGİ PANELİ ---
    try:
        # HATA KORUMASI 3: Scalar Dönüşüm (.item() kullanımı)
        current_price = df['Close'].iloc[-1].item()
        prev_price = df['Close'].iloc[-2].item()
        degisim = ((current_price - prev_price) / prev_price) * 100
        
        rsi_now = df['RSI'].iloc[-1].item()
        atr_now = df['ATR'].iloc[-1].item()
        
        stock_name = info.get('longName', ticker)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Enstrüman", stock_name)
        c2.metric("Son Fiyat", f"{current_price:.2f}", f"%{degisim:.2f}")
        
        rsi_color = "normal"
        if rsi_now > 70: rsi_durum = "Aşırı Alım 🔴"
        elif rsi_now < 30: rsi_durum = "Aşırı Satım 🟢"
        else: rsi_durum = "Nötr ⚪"
        
        c3.metric("RSI Momentum", f"{rsi_now:.1f}", rsi_durum)
        c4.metric("Volatilite (Risk)", f"±{atr_now:.2f}")
        
    except Exception as e:
        st.error(f"Veri işleme hatası: {e}")
        st.stop()

    st.markdown("---")

    # --- GRAFİK ---
    st.subheader("📊 Profesyonel Fiyat Grafiği")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Fiyat'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='#00cc96', width=1), name='SMA 50'))
    
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- LSTM YAPAY ZEKA MODÜLÜ ---
    st.subheader("🧠 Yapay Zeka (LSTM) Tahmini")
    
    with st.spinner("Sinir ağları (Neural Networks) eğitiliyor..."):
        try:
            # 1. Veri Hazırlığı
            data = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Tahmin için son 60 günü al
            X_input = scaled_data[-60:].reshape(1, 60, 1)
            
            # 2. Model Mimarisi
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # 3. Hızlı Eğitim (Smart Training)
            # Sadece son 1 yılın verisiyle eğiterek canlı kullanımda hız kazandırıyoruz
            train_window = 252 # Ortalama 1 borsa yılı
            if len(scaled_data) > train_window:
                train_data = scaled_data[-train_window:]
            else:
                train_data = scaled_data
                
            X_train, y_train = [], []
            for i in range(60, len(train_data)):
                X_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Eğitimi Başlat
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # 4. Gelecek Tahmini
            prediction_scaled = model.predict(X_input)
            prediction = float(scaler.inverse_transform(prediction_scaled)[0][0])
            
            # 5. Sonuç Hesaplama ve Gösterim
            ai_degisim_yuzde = ((prediction - current_price) / current_price) * 100
            
            col_ai1, col_ai2 = st.columns([1, 2])
            
            with col_ai1:
                renk = "normal" if prediction > current_price else "inverse"
                st.metric("1 Ay Sonraki AI Hedefi", f"{prediction:.2f} TL", f"%{ai_degisim_yuzde:.2f}", delta_color=renk)
                
            with col_ai2:
                if prediction > current_price:
                    stop_loss = current_price - (atr_now * 1.5)
                    st.success(f"🚀 **YÜKSELİŞ SİNYALİ:** Yapay zeka trendin yukarı olduğunu öngörüyor.\n\n🛡️ **Güvenli Stop-Loss:** {stop_loss:.2f} TL")
                else:
                    stop_loss = current_price + (atr_now * 1.5)
                    st.error(f"📉 **DÜŞÜŞ/BEKLE SİNYALİ:** Yapay zeka fiyatın gevşeyebileceğini öngörüyor.\n\n🛡️ **Short Stop-Loss:** {stop_loss:.2f} TL")
                    
        except Exception as e:
            st.warning(f"AI Modeli çalışırken teknik bir sorun oluştu: {e}")

    # --- BİLGİ ---
    st.divider()
    st.caption("⚠️ Yasal Uyarı: Bu yazılım bir mühendislik çalışmasıdır. Veriler Yahoo Finance üzerinden sağlanır ve 15dk gecikmeli olabilir. Yatırım tavsiyesi değildir.")
