import pandas as pd
import numpy as np
import yfinance as yf
from neuralforecast import NeuralForecast
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
STOCK = "TATASTEEL.NS"
MODEL_PATH = "./tata_swing_models"

# ==========================================
# 1. GET RECENT DATA
# ==========================================
print(f"📊 Fetching recent data for {STOCK}...")

def get_data(symbol):
    data = yf.download(symbol, period="5y", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    
    # --- ALL FEATURES (Must match training) ---
    # Price Transforms
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility_5'] = data['Log_Returns'].rolling(5).std()
    
    # Volume
    data['Vol_MA20'] = data['Volume'].rolling(20).mean()
    data['Vol_Ratio'] = data['Volume'] / data['Vol_MA20']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Width
    bb_mid = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Width'] = ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std)) / bb_mid
    
    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    
    data.dropna(inplace=True)
    return data

df_raw = get_data(STOCK)

# Prepare for NeuralForecast with ALL features
EXOG_VARS = ['RSI', 'MACD', 'MACD_Hist', 'Vol_Ratio', 'BB_Width', 'Momentum_5', 'Log_Returns', 'Volatility_5']

ai_df = pd.DataFrame({
    'unique_id': 'TATA', 
    'ds': df_raw['Date'], 
    'y': df_raw['Close']
})

# Add all exogenous variables
for col in EXOG_VARS:
    ai_df[col] = df_raw[col].values

# ==========================================
# 2. LOAD PRE-TRAINED MODEL
# ==========================================
print(f"🤖 Loading model from {MODEL_PATH}...")

nf = NeuralForecast.load(path=MODEL_PATH)

# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================
print("🔮 Making predictions for next 5 days...")

forecast = nf.predict(df=ai_df)

# Ensemble average
model_cols = [c for c in forecast.columns if 'NHITS' in c]
forecast['Ensemble'] = forecast[model_cols].mean(axis=1)

# Calculate signal strength
last_price = df_raw['Close'].iloc[-1]
forecast['Signal_Pct'] = (forecast['Ensemble'] - last_price) / last_price * 100

print("\n" + "="*60)
print("📈 PREDICTIONS FOR NEXT 5 DAYS:")
print("="*60)
print(forecast[['ds', 'Ensemble', 'Signal_Pct']].to_string(index=False))

print("\n" + "="*60)
print(f"💡 Current Price: ₹{last_price:.2f}")
print(f"📊 Average Predicted Change: {forecast['Signal_Pct'].mean():.2f}%")
print(f"🎯 Target Price (Day 5): ₹{forecast['Ensemble'].iloc[-1]:.2f}")
print("="*60)

# Trading Signal
avg_signal = forecast['Signal_Pct'].mean()
if avg_signal > 1.0:
    print("\n✅ STRONG BUY SIGNAL - Model predicts upward trend")
elif avg_signal > 0.1:
    print("\n📈 WEAK BUY SIGNAL - Model predicts slight upward trend")
elif avg_signal < -1.0:
    print("\n❌ SELL SIGNAL - Model predicts downward trend")
else:
    print("\n⏸️  HOLD - No strong signal")