"""
Test Suite for Reinforcement Learning Agent System
Tests the RL agent for Phase 2 Day 16
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.ai.reinforcement_learning import (
    DQNAgent, TradingEnvironment, RLTrainer, PrioritizedReplayBuffer,
    AgentConfig, AgentType, ActionType, RewardType, TradingState, TradingAction,
    RewardComponents, Experience, create_default_agent_config, create_trading_environment
)


class TestReinforcementLearning:
    """Test Reinforcement Learning Agent System"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        
        # Generate realistic price data
        base_price = 2000.0
        returns = np.random.normal(0, 0.02, 1000)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'volume': np.random.uniform(1000, 10000, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'macd': np.random.normal(0, 2, 1000),
            'bb_upper': [p * 1.02 for p in prices],
            'bb_lower': [p * 0.98 for p in prices],
            'sma_20': prices,  # Simplified
            'sma_50': prices   # Simplified
        })
        
        return data
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing"""
        return AgentConfig(
            agent_type=AgentType.DQN,
            state_size=95,
            action_size=7,
            learning_rate=0.01,  # Higher for faster testing
            gamma=0.9,
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99,
            memory_size=1000,  # Smaller for testing
            batch_size=16,     # Smaller for testing
            target_update_freq=10,
            hidden_layers=[32, 16],  # Smaller for testing
            use_prioritized_replay=False,  # Simpler for testing
            use_double_dqn=False,
            use_dueling=False
        )
    
    @pytest.fixture
    def trading_environment(self, sample_market_data):
        """Create trading environment for testing"""
        return TradingEnvironment(sample_market_data.iloc[:100], initial_balance=10000.0)
    
    def test_agent_config_creation(self):
        """Test AgentConfig creation"""
        config = AgentConfig(agent_type=AgentType.DQN)
        
        assert config.agent_type == AgentType.DQN
        assert config.state_size == 50  # Default value
        assert config.action_size == 7
        assert isinstance(config.hidden_layers, list)
        assert config.reward_type == RewardType.RISK_ADJUSTED
    
    def test_default_agent_config(self):
        """Test default agent configuration creation"""
        config = create_default_agent_config()
        
        assert config.agent_type == AgentType.DDQN
        assert config.state_size == 95
        assert config.action_size == len(ActionType)
        assert config.use_prioritized_replay
        assert config.use_double_dqn
        assert config.use_dueling
    
    def test_trading_state_creation(self):
        """Test TradingState creation and conversion"""
        price_features = np.random.random((20, 4))
        tech_indicators = np.random.random(6)
        position_info = np.random.random(4)
        portfolio_metrics = np.random.random(4)
        market_regime = 1
        
        state = TradingState(
            price_features=price_features,
            technical_indicators=tech_indicators,
            position_info=position_info,
            portfolio_metrics=portfolio_metrics,
            market_regime=market_regime,
            timestamp=datetime.now()
        )
        
        state_array = state.to_array()
        expected_size = 20*4 + 6 + 4 + 4 + 1  # 95
        assert state_array.shape == (expected_size,)
        assert isinstance(state_array, np.ndarray)
    
    def test_trading_action_creation(self):
        """Test TradingAction creation"""
        action = TradingAction(
            action_type=ActionType.BUY,
            size_fraction=0.2,
            confidence=0.8
        )
        
        assert action.action_type == ActionType.BUY
        assert action.size_fraction == 0.2
        assert action.confidence == 0.8
        assert isinstance(action.metadata, dict)
    
    def test_reward_components(self):
        """Test RewardComponents structure"""
        components = RewardComponents(
            profit_reward=0.1,
            risk_penalty=-0.05,
            drawdown_penalty=-0.02,
            transaction_cost=-0.001,
            sharpe_bonus=0.01
        )
        
        assert components.profit_reward == 0.1
        assert components.risk_penalty == -0.05
        assert components.total_reward == 0.0  # Default
    
    def test_prioritized_replay_buffer(self):
        """Test PrioritizedReplayBuffer functionality"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Test adding experiences
        for i in range(50):
            experience = Experience(
                state=np.random.random(10),
                action=i % 7,
                reward=np.random.random(),
                next_state=np.random.random(10),
                done=False
            )
            buffer.add(experience, priority=np.random.random())
        
        assert len(buffer) == 50
        
        # Test sampling
        experiences, indices, weights = buffer.sample(batch_size=10)
        assert len(experiences) == 10
        assert len(indices) == 10
        assert len(weights) == 10
        
        # Test priority updates
        new_priorities = np.random.random(10)
        buffer.update_priorities(indices, new_priorities)
    
    def test_trading_environment_creation(self, sample_market_data):
        """Test TradingEnvironment creation"""
        env = TradingEnvironment(sample_market_data.iloc[:100], initial_balance=10000.0)
        
        assert env.initial_balance == 10000.0
        assert len(env.data) == 100
        assert env.action_space.n == len(ActionType)
        assert env.observation_space.shape == (95,)
    
    def test_trading_environment_reset(self, trading_environment):
        """Test environment reset functionality"""
        initial_state = trading_environment.reset()
        
        assert isinstance(initial_state, TradingState)
        assert trading_environment.current_step == 0
        assert trading_environment.balance == trading_environment.initial_balance
        assert trading_environment.position == 0.0
        assert trading_environment.total_trades == 0
    
    def test_trading_environment_step(self, trading_environment):
        """Test environment step functionality"""
        state = trading_environment.reset()
        
        action = TradingAction(
            action_type=ActionType.BUY,
            size_fraction=0.1,
            confidence=0.8
        )
        
        next_state, reward, done, info = trading_environment.step(action)
        
        assert isinstance(next_state, TradingState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'balance' in info
        assert 'position' in info
        assert 'total_trades' in info
    
    def test_trading_environment_buy_action(self, trading_environment):
        """Test buy action execution"""
        trading_environment.reset()
        
        action = TradingAction(
            action_type=ActionType.BUY,
            size_fraction=0.2,
            confidence=1.0
        )
        
        initial_balance = trading_environment.balance
        next_state, reward, done, info = trading_environment.step(action)
        
        assert trading_environment.position > 0  # Should have long position
        assert trading_environment.balance < initial_balance  # Balance should decrease
        assert trading_environment.total_trades == 1
    
    def test_trading_environment_sell_action(self, trading_environment):
        """Test sell action execution"""
        trading_environment.reset()
        
        action = TradingAction(
            action_type=ActionType.SELL,
            size_fraction=0.2,
            confidence=1.0
        )
        
        next_state, reward, done, info = trading_environment.step(action)
        
        assert trading_environment.position < 0  # Should have short position
        assert trading_environment.total_trades == 1
    
    def test_trading_environment_close_long(self, trading_environment):
        """Test close long position"""
        trading_environment.reset()
        
        # First buy
        buy_action = TradingAction(ActionType.BUY, size_fraction=0.2)
        trading_environment.step(buy_action)
        
        # Then close
        close_action = TradingAction(ActionType.CLOSE_LONG)
        next_state, reward, done, info = trading_environment.step(close_action)
        
        assert trading_environment.position == 0.0  # Position should be closed
        assert trading_environment.total_trades == 2
    
    def test_dqn_agent_creation(self, agent_config):
        """Test DQN agent creation"""
        agent = DQNAgent(agent_config)
        
        assert agent.config == agent_config
        assert agent.epsilon == agent_config.epsilon
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert len(agent.episode_rewards) == 0
        assert len(agent.episode_losses) == 0
    
    def test_dqn_agent_standard_network(self, agent_config):
        """Test standard DQN network building"""
        agent_config.use_dueling = False
        agent = DQNAgent(agent_config)
        
        # Test network structure
        assert agent.q_network is not None
        assert agent.q_network.input_shape == (None, agent_config.state_size)
        assert agent.q_network.output_shape == (None, agent_config.action_size)
    
    def test_dqn_agent_dueling_network(self, agent_config):
        """Test dueling DQN network building"""
        agent_config.use_dueling = True
        agent = DQNAgent(agent_config)
        
        # Test network structure
        assert agent.q_network is not None
        assert agent.q_network.input_shape == (None, agent_config.state_size)
        assert agent.q_network.output_shape == (None, agent_config.action_size)
    
    def test_dqn_agent_action_selection(self, agent_config, trading_environment):
        """Test agent action selection"""
        agent = DQNAgent(agent_config)
        state = trading_environment.reset()
        
        # Test training mode (with exploration)
        action = agent.act(state, training=True)
        assert isinstance(action, TradingAction)
        assert action.action_type in ActionType
        assert 0 <= action.size_fraction <= 1
        assert 0 <= action.confidence <= 1
        
        # Test evaluation mode (no exploration)
        agent.epsilon = 0.0  # No exploration
        action = agent.act(state, training=False)
        assert isinstance(action, TradingAction)
    
    def test_dqn_agent_memory(self, agent_config, trading_environment):
        """Test agent memory functionality"""
        agent = DQNAgent(agent_config)
        state = trading_environment.reset()
        
        action = TradingAction(ActionType.BUY, size_fraction=0.1)
        next_state, reward, done, info = trading_environment.step(action)
        
        # Test remembering experience
        initial_memory_size = len(agent.memory)
        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == initial_memory_size + 1
    
    @pytest.mark.slow
    def test_dqn_agent_replay(self, agent_config, trading_environment):
        """Test agent replay training"""
        agent = DQNAgent(agent_config)
        
        # Fill memory with experiences
        state = trading_environment.reset()
        for _ in range(agent_config.batch_size + 5):
            action = agent.act(state, training=True)
            next_state, reward, done, info = trading_environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = trading_environment.reset()
        
        # Test replay
        initial_training_step = agent.training_step
        loss = agent.replay()
        
        assert isinstance(loss, float)
        assert agent.training_step > initial_training_step
    
    def test_dqn_agent_target_network_update(self, agent_config):
        """Test target network update"""
        agent = DQNAgent(agent_config)
        
        # Modify main network weights
        original_weights = agent.q_network.get_weights()
        new_weights = [w + 0.1 for w in original_weights]
        agent.q_network.set_weights(new_weights)
        
        # Update target network
        agent.update_target_network()
        
        # Check that target network has same weights as main network
        target_weights = agent.target_network.get_weights()
        main_weights = agent.q_network.get_weights()
        
        for tw, mw in zip(target_weights, main_weights):
            np.testing.assert_array_equal(tw, mw)
    
    def test_dqn_agent_training_stats(self, agent_config):
        """Test agent training statistics"""
        agent = DQNAgent(agent_config)
        
        # Add some dummy training data
        agent.episode_rewards = [1.0, 2.0, 3.0]
        agent.episode_losses = [0.1, 0.2, 0.3]
        agent.training_step = 100
        
        stats = agent.get_training_stats()
        
        assert 'training_step' in stats
        assert 'epsilon' in stats
        assert 'avg_reward_last_100' in stats
        assert 'avg_loss_last_100' in stats
        assert 'total_episodes' in stats
        assert 'memory_size' in stats
        
        assert stats['training_step'] == 100
        assert stats['total_episodes'] == 3
    
    @pytest.mark.slow
    def test_dqn_agent_save_load(self, agent_config):
        """Test agent save and load functionality"""
        agent = DQNAgent(agent_config)
        
        # Add some training history
        agent.episode_rewards = [1.0, 2.0, 3.0]
        agent.episode_losses = [0.1, 0.2, 0.3]
        agent.training_step = 50
        agent.epsilon = 0.5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_agent")
            
            # Save agent
            agent.save_model(model_path)
            
            # Create new agent and load
            new_agent = DQNAgent(agent_config)
            new_agent.load_model(model_path)
            
            # Check that state was loaded correctly
            assert new_agent.epsilon == 0.5
            assert new_agent.training_step == 50
            assert len(new_agent.episode_rewards) == 3
            assert len(new_agent.episode_losses) == 3
    
    def test_rl_trainer_creation(self, agent_config, trading_environment):
        """Test RL trainer creation"""
        agent = DQNAgent(agent_config)
        trainer = RLTrainer(agent, trading_environment)
        
        assert trainer.agent == agent
        assert trainer.environment == trading_environment
        assert trainer.training_history == []
    
    @pytest.mark.slow
    def test_rl_trainer_training(self, agent_config, trading_environment):
        """Test RL trainer training process"""
        agent = DQNAgent(agent_config)
        trainer = RLTrainer(agent, trading_environment)
        
        # Train for a few episodes
        episodes = 3
        history = trainer.train(episodes, max_steps_per_episode=20)
        
        assert 'episode_rewards' in history
        assert 'episode_losses' in history
        assert 'episode_steps' in history
        assert 'win_rates' in history
        assert 'final_balances' in history
        
        assert len(history['episode_rewards']) == episodes
        assert len(agent.episode_rewards) == episodes
    
    @pytest.mark.slow
    def test_rl_trainer_evaluation(self, agent_config, trading_environment):
        """Test RL trainer evaluation"""
        agent = DQNAgent(agent_config)
        trainer = RLTrainer(agent, trading_environment)
        
        # Quick training
        trainer.train(2, max_steps_per_episode=10)
        
        # Evaluate
        evaluation = trainer.evaluate(episodes=2)
        
        assert 'summary' in evaluation
        assert 'detailed_results' in evaluation
        
        summary = evaluation['summary']
        assert 'avg_return' in summary
        assert 'avg_sharpe' in summary
        assert 'avg_win_rate' in summary
        assert 'success_rate' in summary
    
    def test_action_type_enum(self):
        """Test ActionType enum"""
        assert ActionType.HOLD.value == 0
        assert ActionType.BUY.value == 1
        assert ActionType.SELL.value == 2
        assert ActionType.CLOSE_LONG.value == 3
        assert ActionType.CLOSE_SHORT.value == 4
        assert len(ActionType) == 7
    
    def test_agent_type_enum(self):
        """Test AgentType enum"""
        assert AgentType.DQN.value == "dqn"
        assert AgentType.DDQN.value == "ddqn"
        assert AgentType.DUELING_DQN.value == "dueling_dqn"
        assert AgentType.A3C.value == "a3c"
        assert AgentType.PPO.value == "ppo"
        assert AgentType.SAC.value == "sac"
    
    def test_reward_type_enum(self):
        """Test RewardType enum"""
        assert RewardType.PROFIT_LOSS.value == "profit_loss"
        assert RewardType.SHARPE_RATIO.value == "sharpe_ratio"
        assert RewardType.RISK_ADJUSTED.value == "risk_adjusted"
        assert RewardType.DRAWDOWN_PENALTY.value == "drawdown_penalty"
        assert RewardType.MULTI_OBJECTIVE.value == "multi_objective"
    
    def test_trading_environment_state_size(self, trading_environment):
        """Test trading environment state size calculation"""
        expected_size = 80 + 6 + 4 + 4 + 1  # 95
        actual_size = trading_environment._get_state_size()
        assert actual_size == expected_size
    
    def test_trading_environment_reward_calculation(self, trading_environment):
        """Test reward calculation"""
        components = RewardComponents(
            profit_reward=0.1,
            risk_penalty=-0.05,
            drawdown_penalty=-0.02,
            transaction_cost=-0.001,
            sharpe_bonus=0.01
        )
        
        reward = trading_environment._calculate_reward(components)
        expected = 0.1 - 0.05 - 0.02 - 0.001 + 0.01
        assert abs(reward - expected) < 1e-6
        assert components.total_reward == reward
    
    def test_trading_environment_episode_completion(self, trading_environment):
        """Test episode completion conditions"""
        trading_environment.reset()
        
        # Test completion by reaching end of data
        trading_environment.current_step = len(trading_environment.data) - 1
        action = TradingAction(ActionType.HOLD)
        _, _, done, _ = trading_environment.step(action)
        assert done
        
        # Test completion by balance stop loss
        trading_environment.reset()
        trading_environment.balance = trading_environment.initial_balance * 0.05  # Below 10%
        action = TradingAction(ActionType.HOLD)
        _, _, done, _ = trading_environment.step(action)
        assert done
    
    def test_create_trading_environment_factory(self, sample_market_data):
        """Test trading environment factory function"""
        env = create_trading_environment(sample_market_data, initial_balance=50000.0)
        
        assert isinstance(env, TradingEnvironment)
        assert env.initial_balance == 50000.0
        assert len(env.data) == len(sample_market_data)
    
    def test_experience_namedtuple(self):
        """Test Experience namedtuple"""
        experience = Experience(
            state=np.array([1, 2, 3]),
            action=1,
            reward=0.5,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        
        assert experience.action == 1
        assert experience.reward == 0.5
        assert not experience.done
        assert len(experience.state) == 3
        assert len(experience.next_state) == 3
    
    def test_error_handling_invalid_action(self, trading_environment):
        """Test error handling for invalid actions"""
        trading_environment.reset()
        
        # Try to close long when no position
        action = TradingAction(ActionType.CLOSE_LONG)
        next_state, reward, done, info = trading_environment.step(action)
        
        # Should handle gracefully without error
        assert isinstance(next_state, TradingState)
        assert isinstance(reward, float)
    
    def test_error_handling_empty_memory_replay(self, agent_config):
        """Test error handling for replay with empty memory"""
        agent = DQNAgent(agent_config)
        
        # Try replay with empty memory
        loss = agent.replay()
        assert loss == 0.0  # Should return 0 for empty memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])