"""
Demo for Reinforcement Learning Agent System
Ultimate XAU Super System V4.0 - Phase 2 Day 16

This demo showcases the RL agent capabilities:
1. Agent Creation and Configuration
2. Trading Environment Setup
3. Training Process
4. Performance Evaluation
5. Real-time Trading Simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.core.ai.reinforcement_learning import (
    DQNAgent, TradingEnvironment, RLTrainer,
    AgentConfig, AgentType, ActionType, RewardType,
    create_default_agent_config, create_trading_environment
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_realistic_xau_data(periods: int = 2000) -> pd.DataFrame:
    """Generate realistic XAU price data with technical indicators"""
    print("📊 Generating realistic XAU market data...")
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='h')
    
    # Generate price data with realistic XAU characteristics
    base_price = 2000.0
    trend = 0.0001  # Slight upward trend
    volatility = 0.015  # 1.5% hourly volatility
    
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        # Add trend, mean reversion, and random walk
        price_change = (
            trend +  # Long-term trend
            -0.001 * (prices[-1] - base_price) / base_price +  # Mean reversion
            np.random.normal(0, volatility)  # Random walk
        )
        
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1000))  # Floor price
        
        # Volume inversely correlated with price stability
        volume = np.random.uniform(5000, 15000) * (1 + abs(price_change) * 10)
        volumes.append(volume)
    
    # Add first volume
    volumes.insert(0, 10000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'volume': volumes
    })
    
    # Add technical indicators
    data['returns'] = data['close'].pct_change()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['bb_std'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['sma_20'] + (data['bb_std'] * 2)
    data['bb_lower'] = data['sma_20'] - (data['bb_std'] * 2)
    
    # Moving averages
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['ema_26'] = data['close'].ewm(span=26).mean()
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    print(f"✅ Generated {len(data)} data points")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Average volume: {data['volume'].mean():.0f}")
    
    return data


def demo_agent_creation():
    """Demo 1: Agent Creation and Configuration"""
    print("\n" + "="*60)
    print("🤖 DEMO 1: AGENT CREATION AND CONFIGURATION")
    print("="*60)
    
    # Create different agent configurations
    configs = {
        'Basic DQN': AgentConfig(
            agent_type=AgentType.DQN,
            state_size=95,
            learning_rate=0.001,
            hidden_layers=[128, 64],
            use_prioritized_replay=False,
            use_double_dqn=False,
            use_dueling=False
        ),
        'Double DQN': AgentConfig(
            agent_type=AgentType.DDQN,
            state_size=95,
            learning_rate=0.001,
            hidden_layers=[256, 128],
            use_prioritized_replay=False,
            use_double_dqn=True,
            use_dueling=False
        ),
        'Dueling DDQN': AgentConfig(
            agent_type=AgentType.DUELING_DQN,
            state_size=95,
            learning_rate=0.0005,
            hidden_layers=[256, 128, 64],
            use_prioritized_replay=True,
            use_double_dqn=True,
            use_dueling=True
        )
    }
    
    agents = {}
    
    for name, config in configs.items():
        print(f"\n🔧 Creating {name} Agent...")
        agent = DQNAgent(config)
        agents[name] = agent
        
        print(f"   Agent Type: {config.agent_type.value}")
        print(f"   State Size: {config.state_size}")
        print(f"   Action Size: {config.action_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Hidden Layers: {config.hidden_layers}")
        print(f"   Features: {'Prioritized Replay, ' if config.use_prioritized_replay else ''}"
              f"{'Double DQN, ' if config.use_double_dqn else ''}"
              f"{'Dueling' if config.use_dueling else ''}")
        
        # Show network summary
        print(f"   Network Parameters: {agent.q_network.count_params():,}")
    
    print(f"\n✅ Created {len(agents)} different RL agents")
    return agents


def demo_trading_environment(data: pd.DataFrame):
    """Demo 2: Trading Environment Setup"""
    print("\n" + "="*60)
    print("🏪 DEMO 2: TRADING ENVIRONMENT SETUP")
    print("="*60)
    
    # Create trading environment
    initial_balance = 100000.0
    env = create_trading_environment(data, initial_balance)
    
    print(f"📈 Trading Environment Created:")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   Data Points: {len(env.data)}")
    print(f"   Action Space: {env.action_space.n} actions")
    print(f"   State Space: {env.observation_space.shape[0]} features")
    
    # Test environment functionality
    print(f"\n🔄 Testing Environment Functionality...")
    
    state = env.reset()
    print(f"   Initial State Shape: {state.to_array().shape}")
    print(f"   Initial Balance: ${env.balance:,.2f}")
    print(f"   Initial Position: {env.position}")
    
    # Test different actions
    from src.core.ai.reinforcement_learning import TradingAction
    
    actions_to_test = [
        ('BUY', TradingAction(ActionType.BUY, size_fraction=0.1)),
        ('HOLD', TradingAction(ActionType.HOLD, size_fraction=0.0)),
        ('CLOSE_LONG', TradingAction(ActionType.CLOSE_LONG, size_fraction=0.0))
    ]
    
    for action_name, action in actions_to_test:
        next_state, reward, done, info = env.step(action)
        print(f"   {action_name}: Reward={reward:.4f}, Balance=${info['balance']:,.2f}, "
              f"Position={info['position']:.4f}")
    
    print(f"✅ Environment testing completed")
    return env


def demo_training_process(agent: DQNAgent, env: TradingEnvironment):
    """Demo 3: Training Process"""
    print("\n" + "="*60)
    print("🎯 DEMO 3: TRAINING PROCESS")
    print("="*60)
    
    # Create trainer
    trainer = RLTrainer(agent, env)
    
    print(f"🚀 Starting RL Training...")
    print(f"   Agent: {agent.config.agent_type.value}")
    print(f"   Episodes: 50")
    print(f"   Max Steps per Episode: 200")
    
    # Train the agent
    training_start = datetime.now()
    history = trainer.train(episodes=50, max_steps_per_episode=200)
    training_time = datetime.now() - training_start
    
    print(f"\n✅ Training Completed in {training_time.total_seconds():.1f} seconds")
    
    # Analyze training results
    final_rewards = history['episode_rewards'][-10:]
    final_balances = history['final_balances'][-10:]
    
    print(f"\n📊 Training Results:")
    print(f"   Total Episodes: {len(history['episode_rewards'])}")
    print(f"   Average Reward (last 10): {np.mean(final_rewards):.4f}")
    print(f"   Average Balance (last 10): ${np.mean(final_balances):,.2f}")
    print(f"   Best Episode Reward: {max(history['episode_rewards']):.4f}")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    
    # Plot training progress
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Agent Training Progress', fontsize=16, fontweight='bold')
    
    # Episode rewards
    axes[0, 0].plot(history['episode_rewards'], alpha=0.7)
    axes[0, 0].plot(pd.Series(history['episode_rewards']).rolling(10).mean(), 
                    color='red', linewidth=2, label='10-episode MA')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final balances
    axes[0, 1].plot(history['final_balances'], alpha=0.7, color='green')
    axes[0, 1].axhline(y=env.initial_balance, color='red', linestyle='--', 
                       label='Initial Balance')
    axes[0, 1].set_title('Final Portfolio Balance')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Balance ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Win rates
    axes[1, 0].plot(history['win_rates'], alpha=0.7, color='orange')
    axes[1, 0].set_title('Win Rate per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode losses
    axes[1, 1].plot(history['episode_losses'], alpha=0.7, color='purple')
    axes[1, 1].plot(pd.Series(history['episode_losses']).rolling(10).mean(), 
                    color='red', linewidth=2, label='10-episode MA')
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return history


def demo_performance_evaluation(trainer: RLTrainer):
    """Demo 4: Performance Evaluation"""
    print("\n" + "="*60)
    print("📈 DEMO 4: PERFORMANCE EVALUATION")
    print("="*60)
    
    print("🔍 Evaluating trained agent performance...")
    
    # Evaluate the agent
    evaluation = trainer.evaluate(episodes=20)
    summary = evaluation['summary']
    
    print(f"\n📊 Evaluation Results (20 episodes):")
    print(f"   Average Return: {summary['avg_return']:.2%}")
    print(f"   Return Std Dev: {summary['std_return']:.2%}")
    print(f"   Average Sharpe Ratio: {summary['avg_sharpe']:.2f}")
    print(f"   Average Win Rate: {summary['avg_win_rate']:.2%}")
    print(f"   Average Trades per Episode: {summary['avg_trades']:.1f}")
    print(f"   Success Rate (Positive Return): {summary['success_rate']:.2%}")
    
    # Performance classification
    if summary['avg_return'] > 0.1:
        performance = "🌟 EXCELLENT"
    elif summary['avg_return'] > 0.05:
        performance = "✅ GOOD"
    elif summary['avg_return'] > 0:
        performance = "⚠️ MODERATE"
    else:
        performance = "❌ POOR"
    
    print(f"\n🎯 Performance Rating: {performance}")
    
    # Plot evaluation results
    detailed = evaluation['detailed_results']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Performance Evaluation', fontsize=16, fontweight='bold')
    
    # Returns distribution
    axes[0, 0].hist(detailed['total_returns'], bins=10, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(summary['avg_return'], color='red', linestyle='--', 
                       label=f'Mean: {summary["avg_return"]:.2%}')
    axes[0, 0].set_title('Returns Distribution')
    axes[0, 0].set_xlabel('Total Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe ratios
    axes[0, 1].hist(detailed['sharpe_ratios'], bins=10, alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(summary['avg_sharpe'], color='red', linestyle='--', 
                       label=f'Mean: {summary["avg_sharpe"]:.2f}')
    axes[0, 1].set_title('Sharpe Ratio Distribution')
    axes[0, 1].set_xlabel('Sharpe Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Win rates
    axes[1, 0].hist(detailed['win_rates'], bins=10, alpha=0.7, color='orange')
    axes[1, 0].axvline(summary['avg_win_rate'], color='red', linestyle='--', 
                       label=f'Mean: {summary["avg_win_rate"]:.2%}')
    axes[1, 0].set_title('Win Rate Distribution')
    axes[1, 0].set_xlabel('Win Rate')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total trades
    axes[1, 1].hist(detailed['total_trades'], bins=10, alpha=0.7, color='purple')
    axes[1, 1].axvline(summary['avg_trades'], color='red', linestyle='--', 
                       label=f'Mean: {summary["avg_trades"]:.1f}')
    axes[1, 1].set_title('Total Trades Distribution')
    axes[1, 1].set_xlabel('Total Trades')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return evaluation


def demo_real_time_simulation(agent: DQNAgent, data: pd.DataFrame):
    """Demo 5: Real-time Trading Simulation"""
    print("\n" + "="*60)
    print("⚡ DEMO 5: REAL-TIME TRADING SIMULATION")
    print("="*60)
    
    # Create fresh environment for simulation
    sim_data = data.iloc[-500:].reset_index(drop=True)  # Last 500 points
    env = create_trading_environment(sim_data, initial_balance=100000.0)
    
    print("🎮 Starting Real-time Trading Simulation...")
    print(f"   Data Points: {len(sim_data)}")
    print(f"   Initial Balance: ${env.initial_balance:,.2f}")
    
    # Run simulation
    state = env.reset()
    simulation_log = []
    
    step = 0
    while step < len(sim_data) - 1:
        # Agent makes decision
        action = agent.act(state, training=False)  # No exploration
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Log the step
        current_price = sim_data.iloc[env.current_step-1]['close']
        simulation_log.append({
            'step': step,
            'price': current_price,
            'action': action.action_type.name,
            'size_fraction': action.size_fraction,
            'confidence': action.confidence,
            'balance': info['balance'],
            'position': info['position'],
            'reward': reward,
            'total_trades': info['total_trades']
        })
        
        # Print periodic updates
        if step % 50 == 0:
            print(f"   Step {step}: Price=${current_price:.2f}, "
                  f"Action={action.action_type.name}, "
                  f"Balance=${info['balance']:,.2f}, "
                  f"Position={info['position']:.4f}")
        
        state = next_state
        step += 1
        
        if done:
            break
    
    # Simulation results
    final_balance = simulation_log[-1]['balance']
    total_return = (final_balance - env.initial_balance) / env.initial_balance
    total_trades = simulation_log[-1]['total_trades']
    
    print(f"\n📊 Simulation Results:")
    print(f"   Final Balance: ${final_balance:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Steps Completed: {len(simulation_log)}")
    
    # Performance rating
    if total_return > 0.1:
        rating = "🌟 EXCELLENT"
    elif total_return > 0.05:
        rating = "✅ GOOD"
    elif total_return > 0:
        rating = "⚠️ MODERATE"
    else:
        rating = "❌ POOR"
    
    print(f"   Performance: {rating}")
    
    # Plot simulation results
    sim_df = pd.DataFrame(simulation_log)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Real-time Trading Simulation', fontsize=16, fontweight='bold')
    
    # Price and actions
    axes[0].plot(sim_df['step'], sim_df['price'], label='XAU Price', alpha=0.7)
    
    # Mark buy/sell actions
    buy_points = sim_df[sim_df['action'] == 'BUY']
    sell_points = sim_df[sim_df['action'] == 'SELL']
    
    if not buy_points.empty:
        axes[0].scatter(buy_points['step'], buy_points['price'], 
                       color='green', marker='^', s=50, label='BUY', alpha=0.8)
    if not sell_points.empty:
        axes[0].scatter(sell_points['step'], sell_points['price'], 
                       color='red', marker='v', s=50, label='SELL', alpha=0.8)
    
    axes[0].set_title('Price Chart with Trading Actions')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Portfolio balance
    axes[1].plot(sim_df['step'], sim_df['balance'], color='blue', linewidth=2)
    axes[1].axhline(y=env.initial_balance, color='red', linestyle='--', 
                    label='Initial Balance')
    axes[1].set_title('Portfolio Balance Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Balance ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Position and confidence
    axes[2].plot(sim_df['step'], sim_df['position'], label='Position', alpha=0.7)
    axes[2].plot(sim_df['step'], sim_df['confidence'], label='Confidence', alpha=0.7)
    axes[2].set_title('Position Size and Decision Confidence')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return simulation_log


def main():
    """Main demo function"""
    print("🤖 REINFORCEMENT LEARNING AGENT SYSTEM DEMO")
    print("Ultimate XAU Super System V4.0 - Phase 2 Day 16")
    print("=" * 80)
    
    try:
        # Generate market data
        market_data = generate_realistic_xau_data(periods=2000)
        
        # Demo 1: Agent Creation
        agents = demo_agent_creation()
        
        # Use the best agent (Dueling DDQN) for remaining demos
        best_agent = agents['Dueling DDQN']
        
        # Demo 2: Trading Environment
        trading_env = demo_trading_environment(market_data.iloc[:1500])
        
        # Demo 3: Training Process
        training_history = demo_training_process(best_agent, trading_env)
        
        # Demo 4: Performance Evaluation
        trainer = RLTrainer(best_agent, trading_env)
        trainer.training_history = training_history
        evaluation_results = demo_performance_evaluation(trainer)
        
        # Demo 5: Real-time Simulation
        simulation_log = demo_real_time_simulation(best_agent, market_data)
        
        print("\n" + "="*80)
        print("🎉 REINFORCEMENT LEARNING DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Final summary
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Agents Created: {len(agents)}")
        print(f"   Training Episodes: {len(training_history['episode_rewards'])}")
        print(f"   Evaluation Episodes: 20")
        print(f"   Simulation Steps: {len(simulation_log)}")
        print(f"   Best Agent: Dueling DDQN with Prioritized Replay")
        
        avg_return = evaluation_results['summary']['avg_return']
        print(f"   Average Return: {avg_return:.2%}")
        
        if avg_return > 0.05:
            print("   🌟 Agent shows promising trading performance!")
        else:
            print("   📈 Agent needs more training for optimal performance")
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()