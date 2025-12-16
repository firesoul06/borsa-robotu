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
st.set_page_config(page_title="ProTrader Terminal", layout="wide", page_icon="💹")

# Özel CSS: Profesyonel "Dark Mode" Terminal Görünümü
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-box { border: 1px solid #333; padding: 20px; border-radius: 8px; background-color: #161b22; text-align: center; }
    .news-card { border-left: 4px solid #00cc96; background-color: #161b22; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
    .big-stat { font-size: 24px; font-weight: bold; color: #ffffff; }
    .sub-stat { font-size: 14px; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# --- 2. YARDIMCI FONKSİYONLAR (CACHING & ERROR HANDLING) ---

@st.cache_data(ttl=60) # 60 saniye boyunca veriyi hafızada tut (Hız ve Performans için)
def get_stock_data(symbol, period="2y"):
    try:
        # yfinance veri çekme
        df = yf.download(symbol, period=period, progress=False)
        
        # MultiIndex düzeltme (Kritik)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Veri kontrolü
        if df.empty or 'Close' not in df.columns:
            return None
            
        # Temel İndikatörler
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI Hesapla
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Volatilite) Hesapla
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

def get_company_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        news = stock.news
        return info, news
    except:
        return None, None

# --- 3. ARAYÜZ (SIDEBAR) ---
st.sidebar.title("🛠️ ProTrader Kontrol")
ticker = st.sidebar.text_input("Hisse Kodu", value="THYAO.IS").upper()
period = st.sidebar.selectbox("Veri Aralığı", ["1y", "2y", "5y", "max"], index=1)
run_analysis = st.sidebar.button("ANALİZİ GÜNCELLE 🔄")

st.sidebar.info("""
**İpuçları:**
- BIST için sonuna .IS ekleyin (Örn: ASELS.IS)
- Altın Sertifikası için: ALTIN.IS
- Kripto için: BTC-USD
""")

# --- 4. ANA AKIŞ ---
if run_analysis or ticker: # Sayfa ilk açıldığında veya butona basıldığında çalışır
    
    # Verileri Çek
    with st.spinner('Piyasa verileri, haberler ve bilanço taranıyor...'):
        df = get_stock_data(ticker, period)
        info, news_list = get_company_info(ticker)
    
    # HATA KONTROLÜ
    if df is None:
        st.error(f"⛔ HATA: '{ticker}' verisine ulaşılamadı. Kodun doğru olduğundan ve internet bağlantından emin ol.")
        st.stop()

    # --- ÜST BİLGİ KARTLARI ---
    current_price = float(df['Close'].iloc[-1].item())
    prev_close = float(df['Close'].iloc[-2].item())
    degisim = ((current_price - prev_close) / prev_close) * 100
    
    atr_val = df['ATR'].iloc[-1]
    rsi_val = df['RSI'].iloc[-1]
    
    # Şirket Adı
    long_name = info.get('longName', ticker) if info else ticker
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sembol", long_name)
    with col2:
        st.metric("Son Fiyat", f"{current_price:.2f}", f"%{degisim:.2f}")
    with col3:
        durum = "Aşırı Alım 🔴" if rsi_val > 70 else "Aşırı Satım 🟢" if rsi_val < 30 else "Nötr ⚪"
        st.metric("RSI Momentum", f"{rsi_val:.1f}", durum)
    with col4:
        st.metric("Volatilite (Risk)", f"±{atr_val:.2f}", help="Fiyatın günlük ortalama oynaklığı")

    st.divider()

    # --- 5. PROFESYONEL MUM GRAFİĞİ (CANDLESTICK) ---
    st.subheader("📊 Fiyat Grafiği & Trendler")
    
    fig = go.Figure()
    
    # Mum Grafiği
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Fiyat'))
    
    # Hareketli Ortalamalar
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.5), name='SMA 20 (Kısa Vade)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='blue', width=1.5), name='SMA 50 (Orta Vade)'))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. YAPAY ZEKA TAHMİNİ (Optimize Edilmiş LSTM) ---
    st.subheader("🤖 Yapay Zeka Tahmini (Simülasyon)")
    
    with st.spinner('AI Motoru Çalışıyor...'):
        # Hızlı ve Etkili Model Hazırlığı
        data = df[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Son 60 gün verisi
        X_input = scaled_data[-60:].reshape(1, 60, 1)
        
        # Modeli oluştur (Tekrar tekrar eğitmemek için basit tutuyoruz, gerçek projede model kaydedilir)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Hızlı Eğitim ("Fine Tuning" mantığıyla az epoch)
        # Not: Gerçek zamanlı kullanım için eğitim süresini optimize ettim.
        # Bu kısım her açılışta veriye "overfit" (aşırı uyum) olmaması için dinamik bırakıldı.
        X_train, y_train = [], []
        # Son 1 yıllık veriyi eğitime alıyoruz (Hız için)
        train_len = 250 if len(scaled_data) > 250 else len(scaled_data)
        for i in range(60, train_len):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
            
        if len(X_train) > 0:
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            
            # Tahmin
            prediction_scaled = model.predict(X_input)
            prediction = float(scaler.inverse_transform(prediction_scaled)[0][0])
            
            # Yön ve Sinyal
            ai_degisim = ((prediction - current_price) / current_price) * 100
            
            col_ai1, col_ai2 = st.columns([1, 2])
            
            with col_ai1:
                st.markdown("### 🎯 AI Hedefi")
                if prediction > current_price:
                    st.metric("Tahmin (Kısa Vade)", f"{prediction:.2f}", f"%{ai_degisim:.2f}", delta_color="normal")
                    signal = "AL / TUT 📈"
                    signal_class = "success"
                else:
                    st.metric("Tahmin (Kısa Vade)", f"{prediction:.2f}", f"%{ai_degisim:.2f}", delta_color="inverse")
                    signal = "SAT / BEKLE 📉"
                    signal_class = "error"
            
            with col_ai2:
                st.markdown("### 📢 Strateji Kartı")
                stop_loss_level = current_price - (atr_val * 1.5) if signal == "AL / TUT 📈" else current_price + (atr_val * 1.5)
                
                st.info(f"""
                **Sinyal:** {signal}
                \n**Güvenli Stop-Loss:** {stop_loss_level:.2f}
                \n**Risk Analizi:** Şu anki volatiliteye göre fiyatın **{atr_val:.2f}** puan oynama riski var.
                """)
        else:
            st.warning("Yeterli veri olmadığı için AI tahmini yapılamadı.")

    st.divider()

    # --- 7. SON DAKİKA HABERLERİ & TEMEL VERİLER ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📰 İlgili Haberler")
        if news_list:
            count = 0
            for item in news_list:
                if count >= 5: break # İlk 5 haberi göster
                title = item.get('title', 'Başlık Yok')
                link = item.get('link', '#')
                publisher = item.get('publisher', 'Bilinmiyor')
                # Yayınlanma zamanı (varsa)
                
                st.markdown(f"""
                <div class="news-card">
                    <a href="{link}" target="_blank" style="text-decoration:none; color:white; font-weight:bold;">{title}</a>
                    <br><span style="color:#888; font-size:12px;">{publisher}</span>
                </div>
                """, unsafe_allow_html=True)
                count += 1
        else:
            st.write("Güncel haber bulunamadı.")

    with c2:
        st.subheader("🏢 Şirket Özeti")
        if info:
            mcap = info.get('marketCap', 0)
            pe = info.get('trailingPE', 0)
            
            # Büyük sayıları formatla
            def format_number(num):
                if num > 1_000_000_000: return f"{num/1_000_000_000:.2f} Milyar"
                if num > 1_000_000: return f"{num/1_000_000:.2f} Milyon"
                return f"{num}"

            st.write(f"**Sektör:** {info.get('sector', '-')}")
            st.write(f"**Endüstri:** {info.get('industry', '-')}")
            st.write(f"**Piyasa Değeri:** {format_number(mcap)}")
            st.write(f"**F/K Oranı:** {pe:.2f}")
            st.write(f"**Çalışan Sayısı:** {info.get('fullTimeEmployees', '-')}")
            st.write(f"**Özet:** {info.get('longBusinessSummary', 'Bilgi yok.')[:200]}...")
        else:
            st.write("Temel veriler çekilemedi.")
