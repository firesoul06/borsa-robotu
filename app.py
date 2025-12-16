import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="ProTrade AI Bot", layout="wide", page_icon="📈")

st.markdown("""
<style>
.big-font { font-size:30px !important; font-weight: bold; }
.profit { color: #2ecc71; }
.loss { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ProTrade AI: Çok Değişkenli Borsa Tahmin Botu")
st.markdown("Bu bot; **Fiyat**, **Hacim**, **RSI** ve **Hareketli Ortalamaları** aynı anda analiz eden gelişmiş bir LSTM mimarisi kullanır.")

# --- YAN MENÜ ---
st.sidebar.header("⚙️ Parametreler")
hisse_kodu = st.sidebar.text_input("Hisse Kodu", value="THYAO.IS").upper()
analiz_butonu = st.sidebar.button("🚀 Analizi Başlat")

st.sidebar.info("""
**Nasıl Çalışır?**
Bot sadece fiyata bakmaz. 
1. RSI (Momentum)
2. SMA (Trend)
3. Hacim (İlgi)
verilerini harmanlayarak karar verir.
""")

# --- FİNANSAL GÖSTERGE HESAPLAMALARI ---
def add_technical_indicators(df):
    # 1. RSI Hesaplama (14 günlük)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. SMA (Hareketli Ortalamalar)
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # Kısa vade
    df['SMA_50'] = df['Close'].rolling(window=50).mean() # Orta vade
    
    # Veri kaybı olan ilk satırları (NaN) temizle
    df.dropna(inplace=True)
    return df

# --- VERİ HAZIRLAMA ---
LOOK_BACK = 60
FORECAST_DAYS = 30 

def create_dataset(dataset, look_back=60, forecast_days=30):
    X, y = [], []
    # Çok değişkenli girdi (Features) ama tek çıktı (Close Price)
    for i in range(look_back, len(dataset) - forecast_days):
        X.append(dataset[i-look_back:i, :]) # Tüm özellikleri al
        y.append(dataset[i+forecast_days, 0]) # Sadece Kapanış Fiyatını (0. indeks) hedefle
    return np.array(X), np.array(y)

if analiz_butonu:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("Veriler borsadan çekiliyor...")
        progress_bar.progress(10)
        
        # 1. GELİŞMİŞ VERİ ÇEKME
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365*5) # 5 Yıllık veri (Daha sağlam eğitim için)
        
        df = yf.download(hisse_kodu, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error("Veri bulunamadı! Hisse kodunu kontrol edin.")
            st.stop()
            
        # MultiIndex düzeltme
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Gerekli sütunlar var mı?
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error("Eksik veri sütunları var. Bu hisse teknik analiz için uygun olmayabilir.")
            st.stop()

        # 2. TEKNİK İNDİKATÖRLERİ EKLE
        status_text.text("Teknik indikatörler (RSI, SMA, Volume) hesaplanıyor...")
        df = add_technical_indicators(df)
        progress_bar.progress(30)
        
        # Görselleştirme (Fiyat ve SMA)
        st.subheader(f"📊 {hisse_kodu} Teknik Analiz Grafiği")
        chart_data = df[['Close', 'SMA_20', 'SMA_50']]
        st.line_chart(chart_data)

        # 3. VERİ ÖN İŞLEME (Ölçeklendirme)
        # Modelin kullanacağı özellikler: Close, RSI, SMA_20, SMA_50, Volume
        features = ['Close', 'RSI', 'SMA_20', 'SMA_50', 'Volume']
        data_filtered = df[features].values
        
        # Sadece Kapanış Fiyatı için ayrı bir scaler (Geri dönüşüm için lazım)
        scaler_close = MinMaxScaler(feature_range=(0, 1))
        scaler_close.fit(data_filtered[:, 0].reshape(-1, 1)) # Sadece Close sütunu
        
        # Tüm veriler için genel scaler
        scaler_all = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_all.fit_transform(data_filtered)
        
        # Son güncel verileri sakla
        current_close = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Eğitim setini oluştur
        status_text.text("Yapay Zeka eğitimi için tensörler oluşturuluyor...")
        x_train, y_train = create_dataset(scaled_data, LOOK_BACK, FORECAST_DAYS)
        
        # 4. MODEL MİMARİSİ (PRO SEVİYE)
        # Bidirectional LSTM: Zamanı hem ileri hem geri okur (Daha iyi bağlam kurar)
        model = Sequential()
        model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1)) # Tek çıktı: Fiyat
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early Stopping: Model ezberlemeye başlarsa (overfitting) eğitimi durdur
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        status_text.text("🧠 Nöral Ağlar eğitiliyor... (Bidirectional LSTM)")
        progress_bar.progress(50)
        
        # Eğitimi başlat
        model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=0, callbacks=[early_stop])
        progress_bar.progress(90)
        
        # 5. TAHMİN
        status_text.text("Gelecek simülasyonu yapılıyor...")
        
        # Son 60 günün tüm verilerini (Fiyat, RSI, Hacim vs.) al
        last_60_days = scaled_data[-LOOK_BACK:]
        last_60_days_reshaped = np.reshape(last_60_days, (1, LOOK_BACK, len(features)))
        
        predicted_scaled = model.predict(last_60_days_reshaped)
        
        # Sadece fiyat scaler'ını kullanarak gerçek değere çevir
        tahmin_fiyat = scaler_close.inverse_transform(predicted_scaled)[0][0]
        
        # 6. SONUÇ RAPORU
        progress_bar.progress(100)
        status_text.text("Analiz Tamamlandı!")
        
        degisim = tahmin_fiyat - current_close
        yuzde_degisim = (degisim / current_close) * 100
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mevcut Fiyat", f"{current_close:.2f} TL")
            st.caption(f"RSI Göstergesi: {current_rsi:.2f}")
        
        with col2:
            st.metric("30 Gün Sonraki Hedef", f"{tahmin_fiyat:.2f} TL", f"{degisim:.2f} TL")
            
        with col3:
            if yuzde_degisim > 0:
                st.markdown(f"<span class='big-font profit'>🚀 YÜKSELİŞ BEKLENTİSİ</span>", unsafe_allow_html=True)
                st.markdown(f"**Tahmini Getiri:** %{yuzde_degisim:.2f}")
            else:
                st.markdown(f"<span class='big-font loss'>🔻 DÜŞÜŞ SİNYALİ</span>", unsafe_allow_html=True)
                st.markdown(f"**Tahmini Kayıp:** %{yuzde_degisim:.2f}")

        # RSI Yorumu
        st.write("---")
        st.subheader("🤖 Yapay Zeka Görüşü & Uyarılar")
        
        if current_rsi > 70:
            st.warning("⚠️ **RSI Uyarısı:** Hisse şu an 'Aşırı Alınmış' (Overbought) bölgesinde. Fiyatlar şişmiş olabilir, düzeltme (düşüş) gelme ihtimali yüksek.")
        elif current_rsi < 30:
            st.success("✅ **RSI İpucu:** Hisse 'Aşırı Satılmış' (Oversold) bölgesinde. Bu seviyelerden tepki yükselişi gelebilir.")
        else:
            st.info("ℹ️ **RSI Durumu:** Nötr bölgede. Trend takibi yapılmalı.")
            
        st.error("""
        **YASAL UYARI:** Bu yazılım, karmaşık matematiksel modeller (LSTM) kullanarak geçmiş verilerden örüntü çıkarmaya çalışır. 
        Ancak borsa; haber akışı, savaşlar, siyasi kararlar gibi matematikle ölçülemeyen durumlardan etkilenir. 
        **Bu veriye dayanerek yatırım yaparsanız paranızın tamamını kaybedebilirsiniz.**
        """)

    except Exception as e:
        st.error(f"Sistem Hatası: {e}")
