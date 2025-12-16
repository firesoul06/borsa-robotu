import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go

# --- 1. SİSTEM VE SAYFA AYARLARI ---
st.set_page_config(page_title="ProQuant Ultimate", layout="wide", page_icon="💎")

# Profesyonel Arayüz CSS (Göz yormayan Dark Mode)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 22px; color: #e6edf3; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8b949e; }
    .score-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- 2. GÜVENLİ VERİ MOTORU (DEFENSIVE PROGRAMMING) ---

@st.cache_data(ttl=300) # Temel veriler 5 dk önbellekte
def get_fundamental_data(symbol):
    """Şirket bilançosunu çeker. Hata durumunda programı çökertmez, None döner."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info: return None
        return info
    except:
        return None

@st.cache_data(ttl=60) # Fiyat verileri 1 dk önbellekte
def get_technical_data(symbol, period="2y"):
    """Fiyat verilerini ve indikatörleri hesaplar."""
    try:
        df = yf.download(symbol, period=period, progress=False)
        
        # Kritik Düzeltme: MultiIndex Sütunları
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty or 'Close' not in df.columns: return None
        
        # Veri Temizliği
        df = df.astype(float)
        df.dropna(inplace=True) # Boş verileri temizle
        
        # İndikatörler
        # 1. SMA (Trend)
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. ATR (Risk Yönetimi - Çok Önemli)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        df.dropna(inplace=True) # İndikatör hesaplaması sonrası oluşan NaN'ları temizle
        return df
    except:
        return None

def calculate_score(info):
    """Şirketin finansal sağlığını 100 üzerinden puanlar."""
    if not info: return 0
    score = 0
    checks = 0
    
    # Kriter 1: F/K Oranı (Değerleme)
    pe = info.get('trailingPE')
    if pe is not None:
        checks += 1
        if 0 < pe < 15: score += 25 # Ucuz
        elif 15 <= pe < 30: score += 15 # Normal
    
    # Kriter 2: Borç/Özkaynak (Risk)
    de = info.get('debtToEquity')
    if de is not None:
        checks += 1
        if de < 80: score += 25 # Güvenli
        elif de < 150: score += 10 # Kabul edilebilir
        
    # Kriter 3: Karlılık (ROE)
    roe = info.get('returnOnEquity')
    if roe is not None:
        checks += 1
        if roe > 0.20: score += 25 # Çok iyi
        elif roe > 0.10: score += 15 # İyi
        
    # Kriter 4: Fiyat/Defter Değeri (PD/DD)
    pb = info.get('priceToBook')
    if pb is not None:
        checks += 1
        if pb < 1.5: score += 25
        elif pb < 4: score += 10
        
    if checks == 0: return 0
    # Eksik veri varsa bile mevcut verilerle 100 üzerinden normalize et
    final_score = (score / (checks * 25)) * 100 
    return int(final_score)

# --- 3. YAN MENÜ ---
st.sidebar.header("💎 ProQuant Ultimate")
symbol = st.sidebar.text_input("Hisse Kodu", value="THYAO.IS").upper()
period = st.sidebar.selectbox("Analiz Aralığı", ["1y", "2y", "5y"], index=1)
btn_analiz = st.sidebar.button("ANALİZİ BAŞLAT 🚀")

st.sidebar.info("💡 **İpuçları:**\n- BIST: `THYAO.IS`, `GARAN.IS`\n- Altın: `ALTIN.IS`\n- Kripto: `BTC-USD`")

# --- 4. ANA PROGRAM AKIŞI ---
if btn_analiz or symbol:
    
    with st.spinner("Piyasa verileri işleniyor ve yapay zeka hazırlanıyor..."):
        tech_data = get_technical_data(symbol, period)
        fund_info = get_fundamental_data(symbol)
    
    # Veri Kontrolü (Hata Önleyici)
    if tech_data is None or len(tech_data) < 60:
        st.error("⛔ Yeterli veri bulunamadı. Hisse kodunu kontrol edin veya daha eski bir hisse seçin.")
        st.stop()
        
    # --- A. ÖZET EKRANI ---
    current_price = tech_data['Close'].iloc[-1].item()
    prev_price = tech_data['Close'].iloc[-2].item()
    degisim = ((current_price - prev_price) / prev_price) * 100
    atr_val = tech_data['ATR'].iloc[-1].item()
    rsi_val = tech_data['RSI'].iloc[-1].item()
    
    score = calculate_score(fund_info)
    
    st.title(f"{fund_info.get('longName', symbol) if fund_info else symbol}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Son Fiyat", f"{current_price:.2f}", f"%{degisim:.2f}")
    
    # Dinamik Puan Rengi
    score_color = "#00cc96" if score >= 70 else "#ffa500" if score >= 40 else "#ff4b4b"
    with c2:
        st.markdown(f"**Finansal Sağlık**")
        st.markdown(f'<div class="score-badge" style="background-color:{score_color}">{score}/100</div>', unsafe_allow_html=True)
        
    rsi_durum = "Aşırı Alım 🔴" if rsi_val > 70 else "Aşırı Satım 🟢" if rsi_val < 30 else "Nötr ⚪"
    c3.metric("RSI (Momentum)", f"{rsi_val:.1f}", rsi_durum)
    c4.metric("Risk (ATR)", f"±{atr_val:.2f}")
    
    st.divider()
    
    # --- B. GRAFİK ---
    st.subheader("📈 Teknik Görünüm")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tech_data.index,
                open=tech_data['Open'], high=tech_data['High'],
                low=tech_data['Low'], close=tech_data['Close'], name='Fiyat'))
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- C. YAPAY ZEKA (LSTM) ---
    st.subheader("🧠 Yapay Zeka (LSTM) Tahmini")
    
    with st.spinner("Nöral ağlar eğitiliyor..."):
        try:
            # 1. Veri Hazırlığı
            data = tech_data[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Son 60 gün (Girdi)
            X_input = scaled_data[-60:].reshape(1, 60, 1)
            
            # 2. Model Eğitimi (Hız için son 1 yıl verisi)
            train_window = 252 
            train_data = scaled_data[-train_window:] if len(scaled_data) > train_window else scaled_data
            
            X_train, y_train = [], []
            for i in range(60, len(train_data)):
                X_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Model Mimarisi
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            # Sessiz Eğitim
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # 3. Tahmin
            pred_scaled = model.predict(X_input)
            prediction = float(scaler.inverse_transform(pred_scaled)[0][0])
            ai_change = ((prediction - current_price) / current_price) * 100
            
            # --- D. KARAR MEKANİZMASI (MANTIK KONTROLÜ) ---
            # Stop-Loss Mantığı: Şirket kötüyse (Puan düşükse) stop-loss daha dar olsun (Risk alma!)
            risk_factor = 1.0 if score < 50 else 1.5 if score < 75 else 2.0
            
            col_ai1, col_ai2 = st.columns([1, 2])
            
            with col_ai1:
                color = "normal" if prediction > current_price else "inverse"
                st.metric("1 Ay Sonraki Hedef", f"{prediction:.2f}", f"%{ai_change:.2f}", delta_color=color)
                
            with col_ai2:
                if prediction > current_price:
                    # AI Yükseliş Bekliyor
                    stop_loss = current_price - (atr_val * risk_factor)
                    if score >= 60:
                        st.success(f"🚀 **GÜÇLÜ AL SİNYALİ:** Teknik ve Temel veriler pozitif.\n\n🛡️ Güvenli Stop-Loss: **{stop_loss:.2f}**")
                    else:
                        st.warning(f"⚠️ **RİSKLİ AL SİNYALİ:** AI yükseliş bekliyor AMA şirket puanı düşük ({score}).\n\n🛡️ Dar Stop-Loss: **{stop_loss:.2f}** (Yakın takip et!)")
                else:
                    # AI Düşüş Bekliyor
                    stop_loss = current_price + (atr_val * risk_factor)
                    st.error(f"📉 **SAT / BEKLE:** Trend aşağı yönlü görünüyor.\n\n🛡️ Short Stop-Loss: **{stop_loss:.2f}**")
                    
        except Exception as e:
            st.error(f"AI Modeli Hatası: {e}")
            
    st.divider()
    st.caption("⚠️ Yasal Uyarı: Bu uygulama bir karar destek sistemidir. Yatırım tavsiyesi içermez.")
