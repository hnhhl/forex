"""
Reinforcement Learning Agent System
Advanced AI component for Ultimate XAU Super System V4.0 Phase 2

This module implements sophisticated reinforcement learning agents for:
- Adaptive trading strategy optimization
- Dynamic position sizing and risk management
- Market regime-aware decision making
- Multi-objective reward optimization
- Continuous learning and adaptation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import gym
from gym import spaces
import random
import logging
import threading
import time
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, namedtuple
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class AgentType(Enum):
    """Types of reinforcement learning agents"""
    DQN = "dqn"
    DDQN = "ddqn"
    DUELING_DQN = "dueling_dqn"
    A3C = "a3c"
    PPO = "ppo"
    SAC = "sac"


class ActionType(Enum):
    """Types of trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4
    INCREASE_POSITION = 5
    DECREASE_POSITION = 6


class RewardType(Enum):
    """Types of reward functions"""
    PROFIT_LOSS = "profit_loss"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN_PENALTY = "drawdown_penalty"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class AgentConfig:
    """Configuration for RL agent"""
    agent_type: AgentType
    state_size: int = 50
    action_size: int = 7  # Number of possible actions
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    reward_type: RewardType = RewardType.RISK_ADJUSTED
    use_prioritized_replay: bool = True
    use_double_dqn: bool = True
    use_dueling: bool = True


@dataclass
class TradingState:
    """Current state of the trading environment"""
    price_features: np.ndarray
    technical_indicators: np.ndarray
    position_info: np.ndarray
    portfolio_metrics: np.ndarray
    market_regime: int
    timestamp: datetime
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array"""
        return np.concatenate([
            self.price_features.flatten(),
            self.technical_indicators.flatten(),
            self.position_info.flatten(),
            self.portfolio_metrics.flatten(),
            [self.market_regime]
        ])


@dataclass
class TradingAction:
    """Trading action with parameters"""
    action_type: ActionType
    size_fraction: float = 0.1  # Fraction of portfolio
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardComponents:
    """Components of the reward function"""
    profit_reward: float = 0.0
    risk_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    transaction_cost: float = 0.0
    sharpe_bonus: float = 0.0
    total_reward: float = 0.0


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, experience: Experience, priority: float = None):
        """Add experience to buffer"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class TradingEnvironment:
    """Trading environment for RL agent"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(ActionType))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._get_state_size(),), 
            dtype=np.float32
        )
    
    def reset(self) -> TradingState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.max_balance = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = []
        
        return self._get_current_state()
    
    def step(self, action: TradingAction) -> Tuple[TradingState, float, bool, Dict]:
        """Execute action and return new state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_current_state(), 0.0, True, {}
        
        # Execute action
        reward_components = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(reward_components)
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.balance <= self.initial_balance * 0.1)  # Stop loss
        
        # Prepare info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'reward_components': reward_components
        }
        
        return self._get_current_state(), reward, done, info
    
    def _execute_action(self, action: TradingAction) -> RewardComponents:
        """Execute trading action and return reward components"""
        current_price = self.data.iloc[self.current_step]['close']
        prev_balance = self.balance
        
        reward_components = RewardComponents()
        
        if action.action_type == ActionType.BUY and self.position <= 0:
            # Open long position
            position_size = self.balance * action.size_fraction
            shares = position_size / current_price
            self.position = shares
            self.entry_price = current_price
            self.balance -= position_size
            self.total_trades += 1
            reward_components.transaction_cost = -position_size * 0.001  # 0.1% transaction cost
            
        elif action.action_type == ActionType.SELL and self.position >= 0:
            # Open short position (simplified)
            position_size = self.balance * action.size_fraction
            self.position = -position_size / current_price
            self.entry_price = current_price
            self.total_trades += 1
            reward_components.transaction_cost = -position_size * 0.001
            
        elif action.action_type == ActionType.CLOSE_LONG and self.position > 0:
            # Close long position
            position_value = self.position * current_price
            profit = position_value - (self.position * self.entry_price)
            self.balance += position_value
            if profit > 0:
                self.winning_trades += 1
            reward_components.profit_reward = profit / self.initial_balance
            self.position = 0.0
            self.entry_price = 0.0
            self.total_trades += 1
            
        elif action.action_type == ActionType.CLOSE_SHORT and self.position < 0:
            # Close short position
            position_value = abs(self.position) * self.entry_price
            current_value = abs(self.position) * current_price
            profit = position_value - current_value
            self.balance += position_value + profit
            if profit > 0:
                self.winning_trades += 1
            reward_components.profit_reward = profit / self.initial_balance
            self.position = 0.0
            self.entry_price = 0.0
            self.total_trades += 1
        
        # Update max balance and calculate drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        drawdown = (self.max_balance - self.balance) / self.max_balance
        reward_components.drawdown_penalty = -drawdown * 2.0  # Penalty for drawdown
        
        # Calculate risk penalty based on position size
        position_risk = abs(self.position * current_price) / self.balance if self.balance > 0 else 0
        reward_components.risk_penalty = -position_risk * 0.1 if position_risk > 0.5 else 0
        
        return reward_components
    
    def _calculate_reward(self, components: RewardComponents) -> float:
        """Calculate total reward from components"""
        components.total_reward = (
            components.profit_reward +
            components.risk_penalty +
            components.drawdown_penalty +
            components.transaction_cost +
            components.sharpe_bonus
        )
        return components.total_reward
    
    def _get_current_state(self) -> TradingState:
        """Get current state of the environment"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Price features (last 20 prices)
        start_idx = max(0, self.current_step - 19)
        price_data = self.data.iloc[start_idx:self.current_step + 1]
        
        price_features = np.array([
            price_data['close'].values,
            price_data['high'].values,
            price_data['low'].values,
            price_data['volume'].values if 'volume' in price_data.columns else np.ones(len(price_data))
        ]).T
        
        # Pad if necessary
        if len(price_features) < 20:
            padding = np.zeros((20 - len(price_features), price_features.shape[1]))
            price_features = np.vstack([padding, price_features])
        
        # Technical indicators
        current_data = self.data.iloc[self.current_step]
        tech_indicators = np.array([
            current_data.get('rsi', 50.0),
            current_data.get('macd', 0.0),
            current_data.get('bb_upper', current_data['close']),
            current_data.get('bb_lower', current_data['close']),
            current_data.get('sma_20', current_data['close']),
            current_data.get('sma_50', current_data['close'])
        ])
        
        # Position info
        current_price = current_data['close']
        position_value = self.position * current_price if self.position != 0 else 0
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        elif self.position < 0:
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        
        position_info = np.array([
            self.position,
            position_value / self.initial_balance if self.initial_balance > 0 else 0,
            unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0,
            1.0 if self.position > 0 else (-1.0 if self.position < 0 else 0.0)
        ])
        
        # Portfolio metrics
        total_value = self.balance + position_value
        portfolio_metrics = np.array([
            self.balance / self.initial_balance,
            total_value / self.initial_balance,
            (self.max_balance - total_value) / self.max_balance,  # Current drawdown
            self.winning_trades / max(1, self.total_trades)  # Win rate
        ])
        
        # Market regime (simplified)
        market_regime = 0  # Neutral
        if len(price_data) >= 10:
            recent_returns = price_data['close'].pct_change().dropna()
            if recent_returns.mean() > 0.001:
                market_regime = 1  # Bullish
            elif recent_returns.mean() < -0.001:
                market_regime = -1  # Bearish
        
        return TradingState(
            price_features=price_features,
            technical_indicators=tech_indicators,
            position_info=position_info,
            portfolio_metrics=portfolio_metrics,
            market_regime=market_regime,
            timestamp=datetime.now()
        )
    
    def _get_state_size(self) -> int:
        """Calculate the size of the state vector"""
        # Price features: 20 timesteps * 4 features = 80
        # Technical indicators: 6
        # Position info: 4
        # Portfolio metrics: 4
        # Market regime: 1
        return 80 + 6 + 4 + 4 + 1  # Total: 95


class DQNAgent:
    """Deep Q-Network agent for trading"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory = PrioritizedReplayBuffer(config.memory_size) if config.use_prioritized_replay else deque(maxlen=config.memory_size)
        self.epsilon = config.epsilon
        
        # Build networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        
        logger.info(f"DQN Agent initialized with {config.agent_type.value}")
    
    def _build_network(self) -> keras.Model:
        """Build the neural network"""
        if self.config.use_dueling:
            return self._build_dueling_network()
        else:
            return self._build_standard_network()
    
    def _build_standard_network(self) -> keras.Model:
        """Build standard DQN network"""
        model = models.Sequential([
            layers.Dense(self.config.hidden_layers[0], activation='relu', 
                        input_shape=(self.config.state_size,)),
            layers.Dropout(0.2),
        ])
        
        for units in self.config.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(self.config.action_size, activation='linear'))
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _build_dueling_network(self) -> keras.Model:
        """Build dueling DQN network"""
        input_layer = layers.Input(shape=(self.config.state_size,))
        
        # Shared layers
        x = layers.Dense(self.config.hidden_layers[0], activation='relu')(input_layer)
        x = layers.Dropout(0.2)(x)
        
        for units in self.config.hidden_layers[1:]:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Value stream
        value_stream = layers.Dense(1, activation='linear')(x)
        
        # Advantage stream
        advantage_stream = layers.Dense(self.config.action_size, activation='linear')(x)
        
        # Combine streams using Keras operations
        advantage_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage_stream)
        q_values = layers.Add()([value_stream, layers.Subtract()([advantage_stream, advantage_mean])])
        
        model = models.Model(inputs=input_layer, outputs=q_values)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse'
        )
        
        return model
    
    def act(self, state: TradingState, training: bool = True) -> TradingAction:
        """Choose action based on current state"""
        state_array = state.to_array().reshape(1, -1)
        
        # Epsilon-greedy action selection
        if training and np.random.random() <= self.epsilon:
            action_idx = np.random.choice(self.config.action_size)
            confidence = 0.5  # Low confidence for random actions
        else:
            q_values = self.q_network.predict(state_array, verbose=0)
            action_idx = np.argmax(q_values[0])
            confidence = float(np.max(q_values[0]) / (np.sum(np.abs(q_values[0])) + 1e-8))
        
        # Convert to action
        action_type = ActionType(action_idx)
        
        # Determine position size based on confidence and market conditions
        base_size = 0.1  # Base 10% of portfolio
        size_fraction = base_size * confidence
        
        return TradingAction(
            action_type=action_type,
            size_fraction=size_fraction,
            confidence=confidence
        )
    
    def remember(self, state: TradingState, action: TradingAction, reward: float, 
                next_state: TradingState, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(
            state=state.to_array(),
            action=action.action_type.value,
            reward=reward,
            next_state=next_state.to_array(),
            done=done
        )
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            # Calculate TD error for priority
            state_array = state.to_array().reshape(1, -1)
            next_state_array = next_state.to_array().reshape(1, -1)
            
            current_q = self.q_network.predict(state_array, verbose=0)[0][action.action_type.value]
            next_q = np.max(self.target_network.predict(next_state_array, verbose=0)[0])
            target_q = reward + (self.config.gamma * next_q * (1 - done))
            
            td_error = abs(current_q - target_q)
            self.memory.add(experience, td_error)
        else:
            self.memory.append(experience)
    
    def replay(self) -> float:
        """Train the agent on a batch of experiences"""
        if isinstance(self.memory, PrioritizedReplayBuffer):
            if len(self.memory) < self.config.batch_size:
                return 0.0
            
            experiences, indices, weights = self.memory.sample(self.config.batch_size)
            if not experiences:
                return 0.0
        else:
            if len(self.memory) < self.config.batch_size:
                return 0.0
            
            experiences = random.sample(self.memory, self.config.batch_size)
            weights = np.ones(self.config.batch_size)
            indices = None
        
        # Prepare batch data
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values
        if self.config.use_double_dqn:
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)
            next_q_values = self.target_network.predict(next_states, verbose=0)
            next_q_values = next_q_values[np.arange(len(next_q_values)), next_actions]
        else:
            # Standard DQN
            next_q_values = np.max(self.target_network.predict(next_states, verbose=0), axis=1)
        
        # Calculate target Q values
        target_q_values = current_q_values.copy()
        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.config.gamma * next_q_values[i]
        
        # Train network with importance sampling weights
        sample_weights = weights if isinstance(self.memory, PrioritizedReplayBuffer) else None
        history = self.q_network.fit(
            states, target_q_values,
            batch_size=self.config.batch_size,
            epochs=1,
            verbose=0,
            sample_weight=sample_weights
        )
        
        # Update priorities if using prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            # Calculate new TD errors
            new_q_values = self.q_network.predict(states, verbose=0)
            td_errors = np.abs(target_q_values[np.arange(len(target_q_values)), actions] - 
                              new_q_values[np.arange(len(new_q_values)), actions])
            self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_freq == 0:
            self.update_target_network()
        
        return history.history['loss'][0]
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            self.q_network.save(f"{filepath}_dqn.keras")
            
            # Save agent state
            agent_state = {
                'config': {
                    'agent_type': self.config.agent_type.value,
                    'state_size': self.config.state_size,
                    'action_size': self.config.action_size,
                    'learning_rate': self.config.learning_rate,
                    'gamma': self.config.gamma,
                    'epsilon': self.epsilon,
                    'epsilon_min': self.config.epsilon_min,
                    'epsilon_decay': self.config.epsilon_decay
                },
                'training_step': self.training_step,
                'episode_rewards': self.episode_rewards[-100:],  # Keep last 100
                'episode_losses': self.episode_losses[-100:]
            }
            
            with open(f"{filepath}_agent_state.json", 'w') as f:
                json.dump(agent_state, f, indent=2)
            
            logger.info(f"Model saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.q_network = keras.models.load_model(f"{filepath}_dqn.keras")
            self.target_network = keras.models.load_model(f"{filepath}_dqn.keras")
            
            # Load agent state
            with open(f"{filepath}_agent_state.json", 'r') as f:
                agent_state = json.load(f)
            
            self.epsilon = agent_state['config']['epsilon']
            self.training_step = agent_state['training_step']
            self.episode_rewards = agent_state['episode_rewards']
            self.episode_losses = agent_state['episode_losses']
            
            logger.info(f"Model loaded: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss_last_100': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0,
            'total_episodes': len(self.episode_rewards),
            'memory_size': len(self.memory)
        }


class RLTrainer:
    """Trainer for reinforcement learning agents"""
    
    def __init__(self, agent: DQNAgent, environment: TradingEnvironment):
        self.agent = agent
        self.environment = environment
        self.training_history = []
        
    def train(self, episodes: int, max_steps_per_episode: int = 1000) -> Dict[str, List]:
        """Train the agent"""
        logger.info(f"Starting RL training for {episodes} episodes...")
        
        training_history = {
            'episode_rewards': [],
            'episode_losses': [],
            'episode_steps': [],
            'win_rates': [],
            'final_balances': []
        }
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            total_loss = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > self.agent.config.batch_size:
                    loss = self.agent.replay()
                    total_loss += loss
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Record episode statistics
            self.agent.episode_rewards.append(total_reward)
            self.agent.episode_losses.append(total_loss / max(1, steps))
            
            training_history['episode_rewards'].append(total_reward)
            training_history['episode_losses'].append(total_loss / max(1, steps))
            training_history['episode_steps'].append(steps)
            training_history['win_rates'].append(info.get('win_rate', 0))
            training_history['final_balances'].append(info.get('balance', 0))
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(training_history['episode_rewards'][-10:])
                avg_balance = np.mean(training_history['final_balances'][-10:])
                logger.info(f"Episode {episode + 1}/{episodes} - "
                          f"Avg Reward: {avg_reward:.4f}, "
                          f"Avg Balance: ${avg_balance:.2f}, "
                          f"Epsilon: {self.agent.epsilon:.4f}")
        
        self.training_history = training_history
        logger.info("RL training completed!")
        
        return training_history
    
    def evaluate(self, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        logger.info(f"Evaluating agent for {episodes} episodes...")
        
        evaluation_results = {
            'total_returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': [],
            'total_trades': []
        }
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_returns = []
            
            while True:
                action = self.agent.act(state, training=False)  # No exploration
                next_state, reward, done, info = self.environment.step(action)
                
                episode_returns.append(reward)
                state = next_state
                
                if done:
                    break
            
            # Calculate metrics
            total_return = (info['balance'] - self.environment.initial_balance) / self.environment.initial_balance
            returns_array = np.array(episode_returns)
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
            
            evaluation_results['total_returns'].append(total_return)
            evaluation_results['sharpe_ratios'].append(sharpe_ratio)
            evaluation_results['win_rates'].append(info.get('win_rate', 0))
            evaluation_results['total_trades'].append(info.get('total_trades', 0))
        
        # Calculate summary statistics
        summary = {
            'avg_return': np.mean(evaluation_results['total_returns']),
            'std_return': np.std(evaluation_results['total_returns']),
            'avg_sharpe': np.mean(evaluation_results['sharpe_ratios']),
            'avg_win_rate': np.mean(evaluation_results['win_rates']),
            'avg_trades': np.mean(evaluation_results['total_trades']),
            'success_rate': np.mean([r > 0 for r in evaluation_results['total_returns']])
        }
        
        logger.info(f"Evaluation completed - Avg Return: {summary['avg_return']:.2%}, "
                   f"Avg Sharpe: {summary['avg_sharpe']:.2f}")
        
        return {
            'summary': summary,
            'detailed_results': evaluation_results
        }


# Factory functions
def create_default_agent_config() -> AgentConfig:
    """Create default agent configuration"""
    return AgentConfig(
        agent_type=AgentType.DDQN,
        state_size=95,  # Based on TradingEnvironment state size
        action_size=len(ActionType),
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=100,
        hidden_layers=[256, 128, 64],
        reward_type=RewardType.RISK_ADJUSTED,
        use_prioritized_replay=True,
        use_double_dqn=True,
        use_dueling=True
    )


def create_trading_environment(data: pd.DataFrame, initial_balance: float = 100000.0) -> TradingEnvironment:
    """Create trading environment with data"""
    return TradingEnvironment(data, initial_balance)


if __name__ == "__main__":
    # Example usage
    print("🤖 Reinforcement Learning Agent System")
    print("Phase 2 - AI Systems Expansion")
    
    # Create agent configuration
    config = create_default_agent_config()
    
    # Create agent
    agent = DQNAgent(config)
    
    print(f"✅ RL Agent created: {config.agent_type.value}")
    print(f"   State Size: {config.state_size}")
    print(f"   Action Size: {config.action_size}")
    print(f"   Network: {'Dueling ' if config.use_dueling else ''}{'Double ' if config.use_double_dqn else ''}DQN")
    print("Ready for training and trading!")