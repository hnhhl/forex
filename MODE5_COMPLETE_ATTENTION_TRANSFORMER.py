#!/usr/bin/env python3
"""
🎯 MODE 5.3 COMPLETE: ATTENTION/TRANSFORMER SYSTEM
Ultimate XAU Super System V4.0 → V5.0

Real Transformer Architecture với Custom Financial Attention
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import os
import math

class CompleteAttentionTransformerSystem:
    """Complete Attention/Transformer System với Financial Focus"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1
        }
        self.sequence_length = 60  # 60 bars sequence
        self.d_model = 64  # Model dimension
        self.num_heads = 4  # Multi-head attention
        self.num_layers = 2  # Transformer layers
        self.prediction_horizons = [2, 4, 8]
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            return False
        return True
        
    def positional_encoding(self, length, depth):
        """Create positional encoding for transformer"""
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def create_financial_transformer_block(self, inputs, d_model, num_heads, name_prefix):
        """Create custom financial transformer block"""
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            name=f'{name_prefix}_attention'
        )(inputs, inputs)
        
        # Add & Norm 1
        attention_output = Dropout(0.1, name=f'{name_prefix}_dropout1')(attention_output)
        x1 = LayerNormalization(name=f'{name_prefix}_norm1')(inputs + attention_output)
        
        # Feed Forward Network với Financial adaptations
        ffn_output = Dense(d_model * 4, activation='relu', name=f'{name_prefix}_ffn1')(x1)
        ffn_output = Dropout(0.1, name=f'{name_prefix}_dropout2')(ffn_output)
        ffn_output = Dense(d_model, name=f'{name_prefix}_ffn2')(ffn_output)
        
        # Add & Norm 2
        x2 = LayerNormalization(name=f'{name_prefix}_norm2')(x1 + ffn_output)
        
        return x2
        
    def create_attention_model(self, input_shape):
        """Create complete attention/transformer model"""
        # Input layer
        inputs = Input(shape=input_shape, name='sequence_input')
        
        # Embedding layer để chuyển features thành d_model dimension
        x = Dense(self.d_model, name='input_embedding')(inputs)
        
        # Add positional encoding
        seq_len = input_shape[0]
        pos_encoding = self.positional_encoding(seq_len, self.d_model)
        x = x + pos_encoding
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self.create_financial_transformer_block(
                x, self.d_model, self.num_heads, f'transformer_{i}'
            )
            
        # Global pooling to aggregate sequence information
        x = GlobalAveragePooling1D(name='global_pool')(x)
        
        # Classification head
        x = Dense(32, activation='relu', name='classifier_dense1')(x)
        x = Dropout(0.2, name='classifier_dropout1')(x)
        x = Dense(3, activation='softmax', name='prediction')(x)
        
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Lower learning rate for transformers
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def get_data_and_features(self, timeframe):
        """Get data and calculate advanced features"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 3000)
        if rates is None:
            return None, None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Advanced technical features for transformer
        features = self.calculate_advanced_features(df)
        
        return df, features
        
    def calculate_advanced_features(self, df):
        """Calculate advanced features optimized for attention"""
        # Price features
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility features
        df['bb_upper'] = df['sma_20'] + (df['close'].rolling(20).std() * 2)
        df['bb_lower'] = df['sma_20'] - (df['close'].rolling(20).std() * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Market structure features
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                           (df['low'] > df['low'].shift(1))).astype(int)
        
        # Time features (important for attention)
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['minute'] = df['time'].dt.minute
        
        # Session features
        df['is_asian'] = ((df['hour'] >= 23) | (df['hour'] <= 8)).astype(int)
        df['is_london'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
        
        # Momentum features for attention
        df['momentum_5'] = df['close'].rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        df['momentum_10'] = df['close'].rolling(10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        df['momentum_20'] = df['close'].rolling(20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        
        # Select feature columns
        feature_cols = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'price_change', 'high_low_pct', 'open_close_pct',
            'volume_ratio', 'higher_high', 'lower_low', 'inside_bar',
            'hour', 'day_of_week', 'minute',
            'is_asian', 'is_london', 'is_ny',
            'momentum_5', 'momentum_10', 'momentum_20'
        ]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        return df[feature_cols]
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def create_sequences(self, features, prices, horizon):
        """Create sequences for transformer training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features) - horizon):
            # Get sequence of features
            sequence = features.iloc[i-self.sequence_length:i].values
            
            # Calculate future price movement
            current_price = prices.iloc[i]
            future_price = prices.iloc[i + horizon]
            
            if pd.notna(current_price) and pd.notna(future_price):
                return_pct = (future_price - current_price) / current_price
                
                # Create labels with tighter thresholds for better precision
                if return_pct > 0.001:  # > 0.1%
                    label = 2  # BUY
                elif return_pct < -0.001:  # < -0.1%
                    label = 0  # SELL
                else:
                    label = 1  # HOLD
                    
                X.append(sequence)
                y.append(label)
                
        return np.array(X), np.array(y)
        
    def train_transformer_model(self, tf_name, tf_value, horizon):
        """Train transformer model for specific timeframe and horizon"""
        print(f"🎯 Training Transformer {tf_name}_dir_{horizon}...")
        
        # Get data
        df, features = self.get_data_and_features(tf_value)
        if df is None or len(df) < 1000:
            print(f"❌ Insufficient data for {tf_name}")
            return None, 0
            
        # Create sequences
        X, y = self.create_sequences(features, df['close'], horizon)
        
        if len(X) < 200:
            print(f"❌ Not enough sequences for {tf_name}_dir_{horizon}")
            return None, 0
            
        print(f"  📊 Created {len(X)} sequences with {X.shape[2]} features")
        print(f"  📈 Label distribution: {np.bincount(y)}")
        
        # Normalize features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        model = self.create_attention_model((X.shape[1], X.shape[2]))
        
        print(f"  🏗️ Model Parameters: {model.count_params():,}")
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
        ]
        
        # Training
        history = model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch size for transformers
            epochs=20,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  ✅ {tf_name}_dir_{horizon}: {test_accuracy:.1%} accuracy")
        
        return model, test_accuracy
        
    def run_complete_training(self):
        """Run complete transformer training"""
        print("🎯 MODE 5.3: COMPLETE ATTENTION/TRANSFORMER TRAINING")
        print("=" * 60)
        
        if not self.connect_mt5():
            print("❌ Cannot connect to MT5")
            return {}
            
        all_results = {}
        
        try:
            for tf_name, tf_value in self.timeframes.items():
                print(f"\n📊 Processing {tf_name} timeframe...")
                
                for horizon in self.prediction_horizons:
                    model, accuracy = self.train_transformer_model(tf_name, tf_value, horizon)
                    
                    if model is not None:
                        # Save model
                        os.makedirs('training/xauusdc/models_mode5', exist_ok=True)
                        model_name = f"{tf_name}_dir_{horizon}_transformer"
                        model_path = f"training/xauusdc/models_mode5/{model_name}.h5"
                        model.save(model_path)
                        
                        all_results[model_name] = {
                            'accuracy': float(accuracy),
                            'model_type': 'Transformer',
                            'timeframe': tf_name,
                            'horizon': horizon,
                            'parameters': int(model.count_params()),
                            'sequence_length': self.sequence_length,
                            'd_model': self.d_model,
                            'num_heads': self.num_heads,
                            'num_layers': self.num_layers
                        }
                        
        finally:
            mt5.shutdown()
            
        # Save results
        if all_results:
            results_file = f"mode5_transformer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
        self.print_summary(all_results)
        
        return all_results
        
    def print_summary(self, results):
        """Print training summary"""
        print("\n" + "=" * 60)
        print("🏆 MODE 5.3 TRANSFORMER SUMMARY")
        print("=" * 60)
        
        if not results:
            print("❌ No models trained")
            return
            
        print(f"✅ Transformer models trained: {len(results)}")
        
        # Best models
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print("\n🎯 TOP PERFORMING MODELS:")
        for i, (name, result) in enumerate(sorted_results[:3]):
            params = result['parameters']
            print(f"  {i+1}. {name}: {result['accuracy']:.1%} ({params:,} params)")
            
        # Statistics
        accuracies = [r['accuracy'] for r in results.values()]
        total_params = sum(r['parameters'] for r in results.values())
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  • Best: {max(accuracies):.1%}")
        print(f"  • Average: {np.mean(accuracies):.1%}")
        print(f"  • Baseline (Dense): 84.0%")
        print(f"  • Improvement: +{(max(accuracies) - 0.84)*100:.1f}%")
        
        print(f"\n🏗️ ARCHITECTURE:")
        print(f"  • Total Parameters: {total_params:,}")
        print(f"  • Sequence Length: {self.sequence_length}")
        print(f"  • Model Dimension: {self.d_model}")
        print(f"  • Attention Heads: {self.num_heads}")
        print(f"  • Transformer Layers: {self.num_layers}")
        
        print(f"\n🚀 ADVANTAGES:")
        print(f"  • Self-attention captures dependencies")
        print(f"  • Positional encoding for time awareness")
        print(f"  • Parallel processing capability")
        print(f"  • Excellent for sequence modeling")

if __name__ == "__main__":
    system = CompleteAttentionTransformerSystem()
    results = system.run_complete_training()
    
    if results:
        print("\n🎉 MODE 5.3 COMPLETE!")
        print("📊 Ready for Mode 5.4...")
    else:
        print("\n❌ Training failed!") 