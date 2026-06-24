import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BTC5MinBot:
    def __init__(self, api_key=None, api_secret=None):
        """Initialize the bot with exchange connection"""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.symbol = 'BTC/USDT'
        self.timeframe = '5m'
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_candles(self, limit=500):
        """Fetch historical candles from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands
        df['bb_sma'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action
        df['candle_size'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Target: 1 if next candle closes green (up), 0 if red (down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()
    
    def train_model(self, df):
        """Train the prediction model"""
        feature_cols = ['rsi', 'macd', 'signal', 'macd_hist', 'bb_position', 
                       'atr', 'momentum', 'volume_ratio', 'candle_size', 
                       'body_size', 'upper_wick', 'lower_wick']
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                           random_state=42, n_jobs=-1)
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def predict_probability(self, df):
        """Predict probability for current candle"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        feature_cols = ['rsi', 'macd', 'signal', 'macd_hist', 'bb_position', 
                       'atr', 'momentum', 'volume_ratio', 'candle_size', 
                       'body_size', 'upper_wick', 'lower_wick']
        
        # Get latest candle features
        X_latest = df[feature_cols].iloc[-1:].values
        X_scaled = self.scaler.transform(X_latest)
        
        # Get probability prediction
        proba = self.model.predict_proba(X_scaled)[0]
        up_probability = proba[1] * 100  # Probability of green candle (UP)
        down_probability = proba[0] * 100  # Probability of red candle (DOWN)
        
        return up_probability, down_probability
    
    def print_analysis(self, df, up_prob, down_prob):
        """Print analysis results"""
        latest = df.iloc[-1]
        print("\n" + "="*50)
        print("BTC 5M PREDICTION")
        print("="*50)
        print(f"Timestamp: {latest['timestamp']}")
        print(f"Current Price: ${latest['close']:.2f}")
        print(f"Open: ${latest['open']:.2f} | High: ${latest['high']:.2f} | Low: ${latest['low']:.2f}")
        print("-"*50)
        print(f"UP probability (GREEN 🟢): {up_prob:.2f}%")
        print(f"DOWN probability (RED 🔴): {down_prob:.2f}%")
        print("-"*50)
        
        # Signal
        if up_prob > down_prob:
            signal = f"BULLISH 📈 ({up_prob - down_prob:.2f}% confidence)"
        else:
            signal = f"BEARISH 📉 ({down_prob - up_prob:.2f}% confidence)"
        print(f"Signal: {signal}")
        
        # Technical indicators
        print("-"*50)
        print("Technical Indicators:")
        print(f"  RSI(14): {latest['rsi']:.2f}")
        print(f"  MACD: {latest['macd']:.6f} | Signal: {latest['signal']:.6f}")
        print(f"  Bollinger Position: {latest['bb_position']:.2f}")
        print(f"  ATR(14): {latest['atr']:.4f}")
        print(f"  Momentum: {latest['momentum']:.2f}")
        print(f"  Volume Ratio: {latest['volume_ratio']:.2f}")
        print("="*50 + "\n")
    
    def run(self):
        """Main bot loop"""
        print("Bot starting on Render...")
        print("Initializing BTC 5M prediction bot...")
        
        try:
            # Fetch historical data
            print("Fetching historical data...")
            df = self.fetch_candles(limit=500)
            
            if df is None:
                return
            
            # Calculate indicators
            print("Calculating technical indicators...")
            df = self.calculate_indicators(df)
            
            # Train model
            print("Training prediction model...")
            self.train_model(df)
            
            # Make prediction
            up_prob, down_prob = self.predict_probability(df)
            self.print_analysis(df, up_prob, down_prob)
            
            # Continuous monitoring loop
            print("Bot running... Monitoring BTC 5M candles")
            last_candle_time = df.iloc[-1]['timestamp']
            
            while True:
                try:
                    # Fetch latest data
                    df = self.fetch_candles(limit=500)
                    df = self.calculate_indicators(df)
                    
                    current_candle_time = df.iloc[-1]['timestamp']
                    
                    # Update prediction on new candle
                    if current_candle_time > last_candle_time:
                        # Retrain model with new data
                        self.train_model(df)
                        
                        # Get new prediction
                        up_prob, down_prob = self.predict_probability(df)
                        self.print_analysis(df, up_prob, down_prob)
                        
                        last_candle_time = current_candle_time
                    
                    # Wait before next check (30 seconds)
                    time.sleep(30)
                    
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            print(f"Bot error: {e}")

if __name__ == "__main__":
    # Initialize bot with your API keys (optional for real trading)
    # bot = BTC5MinBot(api_key='your_key', api_secret='your_secret')
    
    # For public data access (no trading)
    bot = BTC5MinBot()
    bot.run()
