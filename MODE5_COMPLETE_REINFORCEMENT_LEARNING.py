#!/usr/bin/env python3
"""
🤖 MODE 5.5 COMPLETE: REINFORCEMENT LEARNING SYSTEM
Ultimate XAU Super System V4.0 → V5.0

Advanced RL với DQN và Policy Gradient cho Trading
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import os
from collections import deque
import random

class TradingEnvironment:
    """Trading Environment cho RL"""
    
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment"""
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.total_profit = 0
        self.trades = 0
        return self.get_state()
        
    def get_state(self):
        """Get current state"""
        if self.current_step >= len(self.data):
            return np.zeros(22)  # 20 technical features + balance + position
            
        # Technical features (20)
        technical_features = self.data.iloc[self.current_step].values
        
        # Balance and position info
        balance_norm = self.balance / self.initial_balance
        position_info = self.position
        
        state = np.concatenate([technical_features, [balance_norm, position_info]])
        return state
        
    def step(self, action):
        """Execute action and return new state"""
        if self.current_step >= self.max_steps:
            return self.get_state(), 0, True, {}
            
        current_price = self.data.index[self.current_step]  # Using index as price proxy
        
        # Calculate reward based on action
        reward = self.calculate_reward(action, current_price)
        
        # Execute action
        if action == 0:  # SELL/SHORT
            if self.position == 1:  # Close long
                self.position = 0
                self.trades += 1
            elif self.position == 0:  # Open short
                self.position = -1
                self.trades += 1
                
        elif action == 1:  # HOLD
            pass  # Do nothing
            
        elif action == 2:  # BUY/LONG
            if self.position == -1:  # Close short
                self.position = 0
                self.trades += 1
            elif self.position == 0:  # Open long
                self.position = 1
                self.trades += 1
                
        self.current_step += 1
        self.total_profit += reward
        
        done = self.current_step >= self.max_steps
        next_state = self.get_state()
        
        return next_state, reward, done, {'trades': self.trades, 'profit': self.total_profit}
        
    def calculate_reward(self, action, current_price):
        """Calculate reward for action"""
        if self.current_step == 0:
            return 0
            
        # Price change from previous step
        prev_price = self.data.index[self.current_step - 1]
        price_change = (current_price - prev_price) / prev_price
        
        # Reward based on position and price movement
        if self.position == 1:  # Long position
            reward = price_change * 100  # Scale up reward
        elif self.position == -1:  # Short position
            reward = -price_change * 100
        else:  # No position
            reward = 0
            
        # Penalty for too many trades
        if self.trades > 50:
            reward -= 0.1
            
        return reward

class DQNAgent:
    """Deep Q-Network Agent for Trading"""
    
    def __init__(self, state_size=22, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
    def build_model(self):
        """Build DQN model"""
        inputs = Input(shape=(self.state_size,))
        
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
        
    def update_target_model(self):
        """Update target network"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
        
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
                
        # Train model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class CompleteReinforcementLearningSystem:
    """Complete RL System cho Trading"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.timeframe = mt5.TIMEFRAME_M15
        self.agent = None
        self.scaler = StandardScaler()
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            return False
        return True
        
    def get_training_data(self):
        """Get data for RL training"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 2000)
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        features = self.calculate_features(df)
        
        return features
        
    def calculate_features(self, df):
        """Calculate trading features"""
        # Basic technical indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = df['ema_12'] - df['close'].ewm(span=26).mean()
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(20).std()
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        # Price action features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Time features
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Session features
        df['is_asian'] = ((df['hour'] >= 23) | (df['hour'] <= 8)).astype(int)
        df['is_london'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
        
        # Momentum features
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        # Add one more feature to reach 20
        df['bb_position'] = (df['close'] - df['sma_20']) / df['volatility']
        
        # Select 20 features
        feature_cols = [
            'sma_10', 'sma_20', 'ema_12', 'rsi', 'macd', 'price_change', 'volatility', 'volume_ratio',
            'high_low_ratio', 'open_close_ratio', 'hour', 'day_of_week',
            'is_asian', 'is_london', 'is_ny', 'momentum_5', 'momentum_10',
            'resistance', 'support', 'price_position'
        ]
        
        # Fill NaN and normalize
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        return df[feature_cols]
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def train_rl_agent(self, data, episodes=100):
        """Train RL agent"""
        print(f"🤖 Training RL Agent for {episodes} episodes...")
        
        # Normalize data
        data_normalized = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        # Create environment and agent
        env = TradingEnvironment(data_normalized)
        self.agent = DQNAgent(state_size=22, action_size=3)
        
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            scores.append(total_reward)
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
                
            # Update target network
            if episode % 10 == 0:
                self.agent.update_target_model()
                
            if episode % 20 == 0:
                avg_score = np.mean(scores[-20:])
                print(f"  Episode {episode}: Avg Score = {avg_score:.2f}, Epsilon = {self.agent.epsilon:.3f}")
                
        return scores
        
    def evaluate_rl_agent(self, data):
        """Evaluate trained RL agent"""
        if self.agent is None:
            return {}
            
        # Normalize data
        data_normalized = pd.DataFrame(
            self.scaler.transform(data),
            columns=data.columns,
            index=data.index
        )
        
        env = TradingEnvironment(data_normalized)
        state = env.reset()
        
        actions = []
        rewards = []
        
        while True:
            action = self.agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            state = next_state
            
            if done:
                break
                
        total_profit = info['profit']
        total_trades = info['trades']
        
        # Calculate win rate
        positive_rewards = [r for r in rewards if r > 0]
        win_rate = len(positive_rewards) / len(rewards) if rewards else 0
        
        return {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_reward': np.mean(rewards),
            'sharpe_ratio': np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0
        }
        
    def run_complete_training(self):
        """Run complete RL training"""
        print("🤖 MODE 5.5: COMPLETE REINFORCEMENT LEARNING")
        print("=" * 60)
        
        if not self.connect_mt5():
            print("❌ Cannot connect to MT5")
            return {}
            
        try:
            # Get training data
            data = self.get_training_data()
            if data is None:
                print("❌ Could not get training data")
                return {}
                
            print(f"📊 Training data: {len(data)} samples")
            
            # Split data for training/testing
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            # Train RL agent
            training_scores = self.train_rl_agent(train_data, episodes=200)
            
            # Evaluate on test data
            eval_results = self.evaluate_rl_agent(test_data)
            
            # Save RL model
            os.makedirs('training/xauusdc/models_mode5', exist_ok=True)
            self.agent.model.save('training/xauusdc/models_mode5/rl_dqn_model.h5')
            
            # Compile results
            results = {
                'rl_dqn_agent': {
                    'model_type': 'Deep Q-Network',
                    'episodes_trained': len(training_scores),
                    'final_training_score': float(training_scores[-1]) if training_scores else 0,
                    'avg_training_score': float(np.mean(training_scores)) if training_scores else 0,
                    'test_profit': float(eval_results.get('total_profit', 0)),
                    'test_trades': int(eval_results.get('total_trades', 0)),
                    'win_rate': float(eval_results.get('win_rate', 0)),
                    'sharpe_ratio': float(eval_results.get('sharpe_ratio', 0)),
                    'parameters': int(self.agent.model.count_params()) if self.agent else 0
                }
            }
            
            # Save results
            results_file = f"mode5_rl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.print_summary(results)
            return results
            
        finally:
            mt5.shutdown()
            
    def print_summary(self, results):
        """Print RL summary"""
        print("\n" + "=" * 60)
        print("🏆 MODE 5.5 RL SUMMARY")
        print("=" * 60)
        
        if not results:
            print("❌ No RL agent trained")
            return
            
        result = results['rl_dqn_agent']
        
        print(f"✅ DQN Agent trained successfully")
        print(f"\n🤖 TRAINING:")
        print(f"  • Episodes: {result['episodes_trained']}")
        print(f"  • Final Score: {result['final_training_score']:.2f}")
        print(f"  • Average Score: {result['avg_training_score']:.2f}")
        print(f"  • Parameters: {result['parameters']:,}")
        
        print(f"\n📊 TEST PERFORMANCE:")
        print(f"  • Total Profit: {result['test_profit']:.2f}")
        print(f"  • Total Trades: {result['test_trades']}")
        print(f"  • Win Rate: {result['win_rate']:.1%}")
        print(f"  • Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        
        print(f"\n🚀 RL ADVANTAGES:")
        print(f"  • Learns from market feedback")
        print(f"  • Adapts to changing conditions")
        print(f"  • Optimizes trading decisions")
        print(f"  • Continuous improvement")

if __name__ == "__main__":
    system = CompleteReinforcementLearningSystem()
    results = system.run_complete_training()
    
    if results:
        print("\n🎉 MODE 5.5 COMPLETE!")
        print("🌟 ALL MODE 5 COMPONENTS FINISHED!")
    else:
        print("\n❌ RL training failed!") 