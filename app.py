import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="QuantAI Pro", layout="wide", page_icon="📊")

# --- CSS İLE PROFESYONEL GÖRÜNÜM ---
st.markdown("""
<style>
    .metric-card { background-color: #0e1117; border: 1px solid #30333F; padding: 20px; border-radius: 10px; text-align: center; }
    .success { color: #00FF7F; font-weight: bold; }
    .danger { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ QuantAI: Profesyonel Algoritmik Alım-Satım Sistemi")
st.markdown("LSTM Derin Öğrenme + Teknik İndikatörler + Backtest Motoru")

# --- YAN MENÜ ---
st.sidebar.header("🛠️ Sistem Parametreleri")
hisse_kodu = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS").upper()
egitim_yili = st.sidebar.selectbox("Veri Seti Büyüklüğü", ["3 Yıl", "5 Yıl", "10 Yıl"], index=1)
epoch = st.sidebar.slider("Eğitim Tekrarı (Epochs)", 20, 100, 30)
analiz_baslat = st.sidebar.button("Sistemi Çalıştır 🚀")

# --- YARDIMCI FONKSİYONLAR (FEATURE ENGINEERING) ---
def add_indicators(df):
    # 1. RSI (Göreceli Güç Endeksi)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD (Trend Takibi)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. Bollinger Bantları (Volatilite)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Upper'] = df['SMA20'] + 2*df['Close'].rolling(window=20).std()
    df['Lower'] = df['SMA20'] - 2*df['Close'].rolling(window=20).std()
    
    # 4. ATR (Average True Range) - Risk Yönetimi İçin
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    df.dropna(inplace=True)
    return df

# --- VERİ SETİ HAZIRLAMA ---
def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i]) # Tüm özellikleri al
        y.append(data[i, 0])         # Sadece Kapanış Fiyatını (indeks 0) hedefle
    return np.array(X), np.array(y)

if analiz_baslat:
    # 1. VERİ ÇEKME
    try:
        yil_dict = {"3 Yıl": 3, "5 Yıl": 5, "10 Yıl": 10}
        start_date = datetime.datetime.now() - datetime.timedelta(days=365*yil_dict[egitim_yili])
        
        with st.spinner('Piyasa verileri çekiliyor ve temizleniyor...'):
            df = yf.download(hisse_kodu, start=start_date, progress=False)
            
            # MultiIndex Düzeltme
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Sütun Kontrolü
            if 'Close' not in df.columns:
                st.error("Veri hatası: 'Close' sütunu bulunamadı.")
                st.stop()
                
            df = add_indicators(df) # İndikatörleri ekle

    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        st.stop()

    # 2. MODEL HAZIRLIĞI
    with st.spinner('Yapay Zeka Mimarisi Kuruluyor...'):
        # Kullanılacak Özellikler: Close, RSI, MACD, Signal, Upper, Lower, ATR
        feature_columns = ['Close', 'RSI', 'MACD', 'Signal', 'Upper', 'Lower', 'ATR']
        data = df[feature_columns].values
        
        # Ölçeklendirme
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Sadece fiyatı geri çevirmek için ayrı scaler
        scaler_price = MinMaxScaler(feature_range=(0, 1))
        scaler_price.fit(data[:, 0].reshape(-1, 1))
        
        LOOK_BACK = 60
        X, y = create_sequences(scaled_data, LOOK_BACK)
        
        # Train/Test Split (%80 Eğitim, %20 Test/Backtest)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

    # 3. MODEL EĞİTİMİ
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model = Sequential()
    # LSTM Katmanları - Daha karmaşık yapı
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    status_text.text(f"Model Eğitiliyor... ({epoch} Epoch)")
    model.fit(X_train, y_train, batch_size=32, epochs=epoch, verbose=0)
    progress_bar.progress(100)
    
    # 4. TAHMİN VE BACKTEST SONUÇLARI
    predictions = model.predict(X_test)
    predictions_inv = scaler_price.inverse_transform(predictions)
    y_test_inv = scaler_price.inverse_transform(y_test.reshape(-1, 1))
    
    # Gelecek Tahmini (Yarını Tahmin Et)
    last_sequence = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, len(feature_columns))
    future_pred = model.predict(last_sequence)
    future_price = scaler_price.inverse_transform(future_pred)[0][0]
    current_price = df['Close'].iloc[-1]
    
    # --- SONUÇLARI GÖRSELLEŞTİRME (PLOTLY) ---
    st.divider()
    
    # KARTLAR (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    degisim = ((future_price - current_price) / current_price) * 100
    
    atr_val = df['ATR'].iloc[-1]
    stop_loss = current_price - (atr_val * 1.5) # ATR bazlı Stop Loss
    take_profit = current_price + (atr_val * 2.0) # Risk/Reward oranı
    
    with col1:
        st.metric("Anlık Fiyat", f"{current_price:.2f} TL")
    with col2:
        st.metric("AI Hedef Fiyat", f"{future_price:.2f} TL", f"%{degisim:.2f}")
    with col3:
        st.metric("Önerilen Stop-Loss", f"{stop_loss:.2f} TL", delta_color="inverse")
    with col4:
        rsi_val = df['RSI'].iloc[-1]
        rsi_status = "Aşırı Alım 🔴" if rsi_val > 70 else "Aşırı Satım 🟢" if rsi_val < 30 else "Nötr ⚪"
        st.metric("RSI Sinyali", f"{rsi_val:.1f}", rsi_status)

    # GRAFİK 1: Gerçek vs Tahmin (Backtest)
    st.subheader("🧪 Backtest Performansı: Yapay Zeka vs Gerçek Piyasa")
    
    test_dates = df.index[train_size+LOOK_BACK:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_inv.flatten(), mode='lines', name='Gerçek Fiyat', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions_inv.flatten(), mode='lines', name='AI Tahmini', line=dict(color='red', width=2, dash='dot')))
    
    fig.update_layout(title='Modelin Test Verisi Üzerindeki Performansı', xaxis_title='Tarih', yaxis_title='Fiyat', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    # TİCARET SİNYALİ
    st.subheader("📢 Profesyonel İşlem Stratejisi")
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        if degisim > 0:
            st.success(f"🟢 **AL (LONG) Sinyali:** Model yükseliş öngörüyor. Ancak RSI {rsi_val:.1f} seviyesinde.")
            st.write(f"- **Giriş:** {current_price:.2f} TL")
            st.write(f"- **Hedef (TP):** {take_profit:.2f} TL")
            st.write(f"- **Zarar Kes (SL):** {stop_loss:.2f} TL")
        else:
            st.error(f"🔴 **SAT (SHORT) / BEKLE Sinyali:** Model düşüş öngörüyor.")
            st.write(f"- Piyasa yönü aşağı. Nakitte kalmak veya açığa satış düşünmek daha mantıklı olabilir.")
            
    with col_r:
        # Hata Oranı (RMSE)
        rmse = np.sqrt(np.mean(((predictions_inv - y_test_inv) ** 2)))
        st.info(f"📊 Model Hata Payı (RMSE): **±{rmse:.2f} TL**")
        st.caption("Bu değer, yapay zekanın ortalama kaç TL yanıldığını gösterir. Düşük olması iyidir.")

    st.warning("⚠️ Yasal Uyarı: Bu bir simülasyondur. Gerçek parayla işlem yapmadan önce mutlaka profesyonel danışmanlık alınız.")
