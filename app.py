import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime

# Sayfa Ayarları
st.set_page_config(page_title="AI Borsa Kahini", layout="centered")

st.title("📈 AI Destekli Borsa Tahmin Botu (LSTM)")
st.write("İstediğiniz hisse senedini girin, yapay zeka geçmiş verileri öğrenip gelecek tahmini yapsın.")

# Yan menü (Sidebar) ayarları
st.sidebar.header("Ayarlar")
hisse_kodu = st.sidebar.text_input("Hisse Kodu (Örn: THYAO.IS, AAPL)", value="THYAO.IS").upper()
epoch_sayisi = st.sidebar.slider("Eğitim Turu (Epoch)", min_value=10, max_value=50, value=20, step=5)
analiz_butonu = st.sidebar.button("Analizi Başlat")

# Sabit Değişkenler
LOOK_BACK = 60
FORECAST_DAYS = 30 

def create_dataset(dataset, look_back=60, forecast_days=30):
    X, y = [], []
    for i in range(look_back, len(dataset) - forecast_days):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i+forecast_days, 0])
    return np.array(X), np.array(y)

if analiz_butonu:
    st.info(f"{hisse_kodu} için veriler indiriliyor ve model eğitiliyor. Lütfen bekleyin...")
    
    # İlerleme çubuğu ve spinner
    with st.spinner('Yapay Zeka hisse hareketlerini öğreniyor... (Bu işlem 30-60 sn sürebilir)'):
        
        # 1. VERİ ÇEKME
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365*4) # 4 yıllık veri
        
        try:
            df = yf.download(hisse_kodu, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("Veri bulunamadı! Hisse kodunu doğru girdiğinizden emin olun (BIST için sonuna .IS ekleyin).")
            else:
                # Veriyi Görselleştirme
                st.subheader("📊 Son 4 Yıllık Fiyat Grafiği")
                st.line_chart(df['Close'])
                
                # Veri Hazırlığı
                data = df.filter(['Close'])
                dataset = data.values
                
                # --- DÜZELTİLEN KISIM BURASI ---
                # Eskiden: float(dataset[-1]) hata veriyordu.
                # Şimdi: dataset[-1][0] veya .item() ile içindeki net sayıyı alıyoruz.
                current_price = float(dataset[-1].item()) 
                # -------------------------------
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                
                x_train, y_train = create_dataset(scaled_data, LOOK_BACK, FORECAST_DAYS)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                # 2. MODEL EĞİTİMİ
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=32, epochs=epoch_sayisi, verbose=0)
                
                # 3. TAHMİN
                last_days = scaled_data[-LOOK_BACK:]
                last_days_reshaped = np.reshape(last_days, (1, LOOK_BACK, 1))
                predicted_price_scaled = model.predict(last_days_reshaped)
                
                # inverse_transform [1,1] boyutunda döner, [0][0] ile sayıyı alırız
                tahmin_fiyat = float(scaler.inverse_transform(predicted_price_scaled)[0][0])
                
                # 4. SONUÇ GÖSTERİMİ
                degisim = tahmin_fiyat - current_price
                yuzde_degisim = (degisim / current_price) * 100
                
                st.divider()
                st.subheader("🔮 30 Gün Sonraki Tahmin")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Şu Anki Fiyat", value=f"{current_price:.2f}")
                
                with col2:
                    st.metric(label="Tahmini Fiyat", value=f"{tahmin_fiyat:.2f}", delta=f"{degisim:.2f}")
                    
                with col3:
                    if yuzde_degisim > 0:
                        st.success(f"Yükseliş Bekleniyor: %{yuzde_degisim:.2f}")
                    else:
                        st.error(f"Düşüş Bekleniyor: %{yuzde_degisim:.2f}")
                
                st.warning("⚠️ YASAL UYARI: Bu proje sadece eğitim amaçlıdır ve yapay zeka denemesi niteliğindedir. Asla yatırım tavsiyesi olarak değerlendirilmemelidir.")
                
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
