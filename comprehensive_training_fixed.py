import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
from core.shared.unified_feature_engine import UnifiedFeatureEngine

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/comprehensive_training_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_all_data():
    """Load and combine all available data"""
    logger = logging.getLogger(__name__)
    logger.info("Loading all available data...")
    
    all_data = []
    data_sources = []
    
    # 1. Load MT5 Maximum Data (High quality)
    mt5_path = "data/maximum_mt5_v2"
    if os.path.exists(mt5_path):
        logger.info("Loading MT5 Maximum Data...")
        for file in os.listdir(mt5_path):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(mt5_path, file)
                    df = pd.read_csv(file_path)
                    
                    # Standardize columns
                    if 'time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['time'])
                    
                    # Add metadata
                    timeframe = file.split('_')[1]  # Extract M1, H1, etc.
                    df['source'] = 'MT5_Maximum'
                    df['timeframe'] = timeframe
                    
                    all_data.append(df)
                    data_sources.append(f"MT5_{timeframe}: {len(df)} records")
                    logger.info(f"  Loaded {file}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"  Error loading {file}: {e}")
    
    # 2. Load Working Free Data (High volume) - Limited selection
    working_path = "data/working_free_data"
    if os.path.exists(working_path):
        logger.info("Loading Working Free Data...")
        priority_files = ['XAUUSD_H1_realistic.csv', 'XAUUSD_H4_realistic.csv', 'XAUUSD_D1_realistic.csv']
        
        for priority_file in priority_files:
            file_path = os.path.join(working_path, priority_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Standardize columns
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    
                    # Add metadata
                    timeframe = priority_file.split('_')[1]  # Extract H1, H4, D1
                    df['source'] = 'Working_Free'
                    df['timeframe'] = timeframe
                    
                    all_data.append(df)
                    data_sources.append(f"Working_{timeframe}: {len(df)} records")
                    logger.info(f"  Loaded {priority_file}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"  Error loading {priority_file}: {e}")
    
    if not all_data:
        raise Exception("No data loaded! Check data directories.")
    
    # Combine all data
    logger.info("Combining all datasets...")
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Remove duplicates based on datetime if possible
    if 'datetime' in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='first')
        dedup_count = len(combined_df)
        logger.info(f"Removed {initial_count - dedup_count} duplicates")
    
    logger.info(f"Total combined data: {len(combined_df):,} records")
    logger.info("Data sources:")
    for source in data_sources:
        logger.info(f"  - {source}")
    
    return combined_df

def prepare_comprehensive_training_data(df):
    """Prepare training data with unified features"""
    logger = logging.getLogger(__name__)
    logger.info("Preparing comprehensive training data...")
    
    feature_engine = UnifiedFeatureEngine()
    
    # Find price columns (case insensitive)
    price_columns = ['open', 'high', 'low', 'close']
    volume_columns = ['volume', 'tick_volume', 'Volume']
    
    actual_columns = {}
    for col in price_columns:
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                actual_columns[col] = df_col
                break
    
    # Find volume column
    volume_col = None
    for vol_col in volume_columns:
        if vol_col in df.columns:
            volume_col = vol_col
            break
    
    if len(actual_columns) < 4:  # Need at least OHLC
        raise Exception(f"Missing required columns. Found: {list(actual_columns.keys())}")
    
    logger.info(f"Using columns: {actual_columns}")
    if volume_col:
        logger.info(f"Volume column: {volume_col}")
    
    # Create standardized DataFrame
    clean_df = pd.DataFrame()
    clean_df['open'] = pd.to_numeric(df[actual_columns['open']], errors='coerce')
    clean_df['high'] = pd.to_numeric(df[actual_columns['high']], errors='coerce')
    clean_df['low'] = pd.to_numeric(df[actual_columns['low']], errors='coerce')
    clean_df['close'] = pd.to_numeric(df[actual_columns['close']], errors='coerce')
    
    if volume_col:
        clean_df['volume'] = pd.to_numeric(df[volume_col], errors='coerce').fillna(1.0)
    else:
        clean_df['volume'] = 1.0  # Default volume
    
    # Add datetime if available
    if 'datetime' in df.columns:
        clean_df['datetime'] = df['datetime']
    
    # Remove invalid data with relaxed validation
    initial_count = len(clean_df)
    
    # Remove NaN values
    clean_df = clean_df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Remove obviously invalid data (but be more lenient)
    clean_df = clean_df[
        (clean_df['high'] >= clean_df['low']) &  # High >= Low
        (clean_df['open'] > 0) &  # Positive prices
        (clean_df['high'] > 0) &
        (clean_df['low'] > 0) &
        (clean_df['close'] > 0) &
        (clean_df['high'] < clean_df['low'] * 10)  # Prevent extreme outliers
    ]
    
    final_count = len(clean_df)
    logger.info(f"Cleaned data: {initial_count:,} -> {final_count:,} records")
    
    if final_count < 1000:
        raise Exception("Insufficient clean data for training")
    
    # Sample data if too large
    if final_count > 100000:
        logger.info(f"Sampling data from {final_count:,} to 100,000 records")
        clean_df = clean_df.sample(n=100000, random_state=42).reset_index(drop=True)
        final_count = 100000
    
    # Generate simplified features first (fallback approach)
    logger.info("Generating simplified features...")
    
    # Calculate basic technical indicators
    clean_df['returns'] = clean_df['close'].pct_change()
    clean_df['price_range'] = (clean_df['high'] - clean_df['low']) / clean_df['close']
    clean_df['body_size'] = abs(clean_df['close'] - clean_df['open']) / clean_df['close']
    
    # Simple moving averages
    clean_df['sma_5'] = clean_df['close'].rolling(window=5).mean()
    clean_df['sma_10'] = clean_df['close'].rolling(window=10).mean()
    clean_df['sma_20'] = clean_df['close'].rolling(window=20).mean()
    
    # Remove rows with NaN from indicators
    clean_df = clean_df.dropna()
    
    # Create feature matrix
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                      'price_range', 'body_size', 'sma_5', 'sma_10', 'sma_20']
    
    X = clean_df[feature_columns].values
    
    logger.info(f"Features shape: {X.shape}")
    
    # Generate targets (next price direction)
    logger.info("Generating training targets...")
    
    close_prices = clean_df['close'].values
    price_changes = np.diff(close_prices)
    
    # Binary classification: 1 if price goes up, 0 if down
    y = (price_changes > 0).astype(int)
    
    # Remove last row from X to match y length
    X = X[:-1]
    
    logger.info(f"Training data prepared:")
    logger.info(f"  Features (X): {X.shape}")
    logger.info(f"  Targets (y): {y.shape}")
    logger.info(f"  Target distribution: UP={np.sum(y)}, DOWN={len(y)-np.sum(y)}")
    
    return X, y

def run_comprehensive_training():
    """Run comprehensive training with all data"""
    logger = setup_logging()
    
    try:
        logger.info("STARTING COMPREHENSIVE TRAINING")
        logger.info("=" * 60)
        
        # Initialize system
        config = SystemConfig()
        config.enable_integrated_training = True
        system = UltimateXAUSystem(config)
        
        if not system.training_system:
            raise Exception("Training system not initialized")
        
        # Load all data
        logger.info("Phase 1: Data Loading")
        combined_data = load_all_data()
        
        # Prepare training data
        logger.info("Phase 2: Data Preparation")
        X, y = prepare_comprehensive_training_data(combined_data)
        
        # Add training data to system
        logger.info("Phase 3: Adding Data to Training System")
        
        # Add data in batches
        batch_size = 1000
        total_added = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            for j in range(len(batch_X)):
                market_data = {
                    'open': float(batch_X[j][0]),
                    'high': float(batch_X[j][1]),
                    'low': float(batch_X[j][2]),
                    'close': float(batch_X[j][3]),
                    'volume': float(batch_X[j][4]),
                    'target': int(batch_y[j])
                }
                
                system.training_system.collect_training_data(market_data)
                total_added += 1
            
            if i % (batch_size * 10) == 0:
                logger.info(f"  Added {total_added:,}/{len(X):,} data points...")
        
        logger.info(f"Total data points added: {total_added:,}")
        
        # Start training
        logger.info("Phase 4: Model Training")
        logger.info("Training 4 unified models: LSTM, CNN, Hybrid, Dense")
        
        training_result = system.start_training()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results/comprehensive_training_{timestamp}.json"
        
        os.makedirs("training_results", exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(training_result, f, indent=2, default=str)
        
        logger.info("COMPREHENSIVE TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Results saved: {results_file}")
        
        # Print summary
        if training_result.get('status') == 'completed':
            logger.info("TRAINING SUMMARY:")
            logger.info(f"  Models trained: {training_result.get('models_trained', 0)}")
            logger.info(f"  Training time: {training_result.get('training_time', 'N/A')}")
            
            if 'results' in training_result:
                for model_name, result in training_result['results'].items():
                    if 'error' not in result:
                        acc = result.get('val_accuracy', 0)
                        logger.info(f"  {model_name.upper()}: {acc:.4f} accuracy")
        
        return training_result
        
    except Exception as e:
        logger.error(f"TRAINING FAILED: {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("AI3.0 COMPREHENSIVE TRAINING (FIXED)")
    print("Training ALL available data with Unified System")
    print("=" * 60)
    
    result = run_comprehensive_training()
    
    if result.get('status') == 'completed':
        print("\nSUCCESS: Comprehensive training completed!")
        print(f"Models trained: {result.get('models_trained', 0)}")
    else:
        print(f"\nFAILED: {result.get('error', 'Unknown error')}") 