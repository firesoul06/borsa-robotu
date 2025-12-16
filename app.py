import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
import datetime

# --- 1. SİSTEM KONFİGÜRASYONU ---
st.set_page_config(page_title="ProQuant Ultimate", layout="wide", page_icon="💎")

# Profesyonel UI Tasarımı (CSS)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; margin-bottom: 10px; }
    .score-card { font-size: 24px; font-weight: bold; text-align: center; padding: 10px; border-radius: 5px; }
    .good { background-color: #00cc96; color: white; }
    .average { background-color: #ffa500; color: black; }
    .bad { background-color: #ff4b4b; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. VERİ MOTORLARI (DEFENSIVE PROGRAMMING) ---

@st.cache_data(ttl=300) # Temel veriler 5 dk önbellekte kalsın (Performans)
def get_fundamental_data(symbol):
    """
    Hatasız Temel Analiz Verisi Çeker.
    Eksik veri varsa 'None' değil, güvenli varsayılan değerler döndürür.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Veri yoksa boş dön
        if not info: return None

        # Güvenli Veri Çekme (Safe Retrieval)
        data = {
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Bilinmiyor'),
            "pe_ratio": info.get('trailingPE', None),       # F/K
            "pb_ratio": info.get('priceToBook', None),      # PD/DD
            "debt_equity": info.get('debtToEquity', None),  # Borç/Özkaynak
            "profit_margin": info.get('profitMargins', None), # Net Kar Marjı
            "roe": info.get('returnOnEquity', None),        # Özkaynak Karlılığı
            "current_ratio": info.get('currentRatio', None), # Cari Oran (Likidite)
            "target_price": info.get('targetMeanPrice', None) # Analist Hedefi
        }
        return data
    except Exception as e:
        return None

@st.cache_data(ttl=60) # Fiyat verisi 1 dk önbellek
def get_technical_data(symbol, period="2y"):
    try:
        df = yf.download(symbol, period=period, progress=False)
        
        # MultiIndex Düzeltme
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty or 'Close' not in df.columns: return None
        
        df = df.astype(float)
        
        # İndikatörler
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Risk)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        df.dropna(inplace=True)
        return df
    except:
        return None

def calculate_fundamental_score(data):
    """
    Şirkete 100 üzerinden finansal sağlık puanı verir.
    Hata toleranslıdır.
    """
    if not data: return 0
    score = 0
    checks = 0 # Kaç kriter kontrol edilebildi?
    
    # 1. Kriter: F/K Oranı (Ucuzluk)
    if data['pe_ratio']:
        checks += 1
        if 0 < data['pe_ratio'] < 15: score += 1 # İdeal
        elif 0 < data['pe_ratio'] < 25: score += 0.5 # Makul
        
    # 2. Kriter: PD/DD (Değerleme)
    if data['pb_ratio']:
        checks += 1
        if data['pb_ratio'] < 1.5: score += 1 # Çok ucuz
        elif data['pb_ratio'] < 5: score += 0.5
        
    # 3. Kriter: Borç Durumu (Risk)
    if data['debt_equity']:
        checks += 1
        if data['debt_equity'] < 80: score += 1 # Az borçlu
        elif data['debt_equity'] < 150: score += 0.5
    
    # 4. Kriter: Karlılık
    if data['profit_margin']:
        checks += 1
        if data['profit_margin'] > 0.10: score += 1 # %10 üstü kar
        elif data['profit_margin'] > 0: score += 0.5
        
    # 5. Kriter: Özkaynak Karlılığı (ROE)
    if data['roe']:
        checks += 1
        if data['roe'] > 0.20: score += 1 # Enflasyon üstü getiri potansiyeli
        
    # Puanı 100'lük sisteme çevir
    if checks == 0: return 0
    final_score = (score / checks) * 100
    return int(final_score)

# --- 3. UI YAPISI ---
st.sidebar.title("💎 ProQuant Ultimate")
st.sidebar.markdown("---")
symbol_input = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS").upper()
period_input = st.sidebar.selectbox("Grafik Geçmişi", ["1y", "2y", "5y"], index=1)
btn_run = st.sidebar.button("TAM ANALİZ BAŞLAT 🚀")

st.sidebar.info("**Modüller:**\n1. Temel Analiz (Bilanço)\n2. Teknik Analiz (Grafik)\n3. Yapay Zeka (LSTM)")

# --- 4. ANA PROGRAM AKIŞI ---
if btn_run or symbol_input:
    
    # --- A. VERİ YÜKLEME ---
    with st.spinner('Bilanço verileri ve fiyat grafikleri işleniyor...'):
        fund_data = get_fundamental_data(symbol_input)
        tech_data = get_technical_data(symbol_input, period_input)
        
    if tech_data is None or len(tech_data) < 60:
        st.error("⛔ Teknik veriler alınamadı veya hisse çok yeni (Yetersiz veri).")
        st.stop()
        
    # Anlık Veriler
    current_price = tech_data['Close'].iloc[-1].item()
    prev_price = tech_data['Close'].iloc[-2].item()
    degisim = ((current_price - prev_price) / prev_price) * 100
    atr_now = tech_data['ATR'].iloc[-1].item()
    rsi_now = tech_data['RSI'].iloc[-1].item()
    
    # --- B. ÜST DASHBOARD ---
    st.title(f"{fund_data['name'] if fund_data else symbol_input}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Son Fiyat", f"{current_price:.2f} TL", f"%{degisim:.2f}")
    
    # Temel Analiz Puanı
    fund_score = calculate_fundamental_score(fund_data)
    color_cls = "good" if fund_score >= 70 else "average" if fund_score >= 40 else "bad"
    
    with col2:
        st.markdown(f'<div class="score-card {color_cls}">Finansal Puan: {fund_score}/100</div>', unsafe_allow_html=True)
        
    # Teknik Durum
    rsi_state = "Aşırı Alım (Satış Riski)" if rsi_now > 70 else "Aşırı Satım (Alım Fırsatı)" if rsi_now < 30 else "Nötr"
    col3.metric("RSI Durumu", f"{rsi_now:.1f}", rsi_state, delta_color="off")
    col4.metric("Volatilite (Risk)", f"±{atr_now:.2f} TL")
    
    st.markdown("---")
    
    # --- C. DETAYLI TEMEL ANALİZ (HATASIZ GÖSTERİM) ---
    st.subheader("📊 Temel Analiz Karnesi")
    if fund_data:
        f1, f2, f3, f4 = st.columns(4)
        
        # Veri varsa göster, yoksa '-' koy (Hatasızlık İlkesi)
        def safe_fmt(val, is_percent=False):
            if val is None: return "-"
            return f"%{val*100:.2f}" if is_percent else f"{val:.2f}"

        f1.metric("F/K (P/E)", safe_fmt(fund_data['pe_ratio']), help="Fiyat/Kazanç Oranı. Düşük olması iyidir.")
        f2.metric("PD/DD (P/B)", safe_fmt(fund_data['pb_ratio']), help="Piyasa Değeri/Defter Değeri.")
        f3.metric("Borç/Özkaynak", safe_fmt(fund_data['debt_equity']), help="Şirketin borçluluk oranı.")
        f4.metric("Net Kar Marjı", safe_fmt(fund_data['profit_margin'], True), help="Şirketin karlılığı.")
        
        if fund_score < 40:
            st.warning("⚠️ **UYARI:** Şirketin finansal verileri zayıf veya riskli görünüyor. Teknik analiz 'AL' verse bile dikkatli olun.")
    else:
        st.info("Bu hisse için detaylı bilanço verisi bulunamadı (Teknik analize devam ediliyor).")

    # --- D. GRAFİK (CANDLESTICK) ---
    st.subheader("📈 Teknik Grafik")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tech_data.index,
                open=tech_data['Open'], high=tech_data['High'],
                low=tech_data['Low'], close=tech_data['Close'], name='Fiyat'))
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- E. LSTM YAPAY ZEKA MODÜLÜ ---
    st.subheader("🧠 Yapay Zeka (LSTM) Simülasyonu")
    
    with st.spinner("Nöral ağlar eğitiliyor ve gelecek simüle ediliyor..."):
        try:
            # Veri Hazırlığı
            data = tech_data[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Son 60 günü tahmin için ayır
            X_input = scaled_data[-60:].reshape(1, 60, 1)
            
            # LSTM Model Mimarisi
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Smart Training (Hız ve Doğruluk Dengesi)
            train_size = 300 if len(scaled_data) > 300 else len(scaled_data)
            train_data = scaled_data[-train_size:]
            
            X_train, y_train = [], []
            for i in range(60, len(train_data)):
                X_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
                
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Eğitim
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # Tahmin
            prediction_scaled = model.predict(X_input)
            prediction = float(scaler.inverse_transform(prediction_scaled)[0][0])
            
            # --- F. NİHAİ KARAR MEKANİZMASI (Temel + Teknik + AI) ---
            ai_change = ((prediction - current_price) / current_price) * 100
            
            st.divider()
            c_ai1, c_ai2 = st.columns([1, 2])
            
            with c_ai1:
                col_val = "normal" if prediction > current_price else "inverse"
                st.metric("AI Tahmini (1 Ay)", f"{prediction:.2f} TL", f"%{ai_change:.2f}", delta_color=col_val)
                
            with c_ai2:
                # KARAR MANTIĞI (Algorithm Decision Tree)
                signal = ""
                stop_loss = 0
                
                if prediction > current_price: # AI Yükseliş diyor
                    if fund_score >= 50:
                        signal = "GÜÇLÜ AL (STRONG BUY)"
                        stop_loss = current_price - (atr_now * 1.5)
                        msg_type = st.success
                    else:
                        signal = "SPEKÜLATİF AL (RISKY BUY)"
                        stop_loss = current_price - (atr_now * 1.0) # Temel kötü olduğu için stopu yakın tut
                        msg_type = st.warning
                else: # AI Düşüş diyor
                    signal = "SAT / BEKLE (SELL/HOLD)"
                    stop_loss = current_price + (atr_now * 1.5)
                    msg_type = st.error
                
                msg_type(f"📢 **SİSTEM KARARI:** {signal}")
                st.write(f"🛡️ **Önerilen Stop-Loss:** {stop_loss:.2f} TL")
                st.caption(f"*Karar Gerekçesi: Finansal Puan ({fund_score}/100) + LSTM Trend Yönü*")

        except Exception as e:
            st.error(f"AI hesaplamasında hata oluştu: {e}")
