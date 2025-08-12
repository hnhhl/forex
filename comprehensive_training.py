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
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_all_data():
    """Load and combine all available data"""
    logger = logging.getLogger(__name__)
    logger.info("🔄 Loading all available data...")
    
    all_data = []
    data_sources = []
    
    # 1. Load MT5 Maximum Data (High quality)
    mt5_path = "data/maximum_mt5_v2"
    if os.path.exists(mt5_path):
        logger.info("📊 Loading MT5 Maximum Data...")
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
                    logger.info(f"  ✅ {file}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error loading {file}: {e}")
    
    # 2. Load Working Free Data (High volume)
    working_path = "data/working_free_data"
    if os.path.exists(working_path):
        logger.info("📊 Loading Working Free Data...")
        for file in os.listdir(working_path):
            if file.endswith('.csv') and 'realistic' in file:
                try:
                    file_path = os.path.join(working_path, file)
                    
                    # Check file size - limit very large files
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    if file_size > 50:  # Limit to 50MB per file
                        logger.info(f"  ⚠️ {file} is large ({file_size:.1f}MB), sampling...")
                        df = pd.read_csv(file_path, nrows=100000)  # Sample 100K rows
                    else:
                        df = pd.read_csv(file_path)
                    
                    # Standardize columns
                    if 'Date' in df.columns and 'Time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    
                    # Add metadata
                    timeframe = file.split('_')[1]  # Extract M1, H1, etc.
                    df['source'] = 'Working_Free'
                    df['timeframe'] = timeframe
                    
                    all_data.append(df)
                    data_sources.append(f"Working_{timeframe}: {len(df)} records")
                    logger.info(f"  ✅ {file}: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error loading {file}: {e}")
    
    if not all_data:
        raise Exception("❌ No data loaded! Check data directories.")
    
    # Combine all data
    logger.info("🔄 Combining all datasets...")
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Remove duplicates based on datetime if possible
    if 'datetime' in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='first')
        dedup_count = len(combined_df)
        logger.info(f"📊 Removed {initial_count - dedup_count} duplicates")
    
    logger.info(f"✅ Total combined data: {len(combined_df):,} records")
    logger.info("📋 Data sources:")
    for source in data_sources:
        logger.info(f"  - {source}")
    
    return combined_df

def prepare_comprehensive_training_data(df):
    """Prepare training data with unified features"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Preparing comprehensive training data...")
    
    feature_engine = UnifiedFeatureEngine()
    
    # Standardize price columns
    price_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Find actual column names (case insensitive)
    actual_columns = {}
    for col in price_columns:
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                actual_columns[col] = df_col
                break
    
    if len(actual_columns) < 4:  # Need at least OHLC
        raise Exception(f"❌ Missing required columns. Found: {list(actual_columns.keys())}")
    
    logger.info(f"📊 Using columns: {actual_columns}")
    
    # Create standardized DataFrame
    clean_df = pd.DataFrame()
    clean_df['open'] = df[actual_columns['open']].astype(float)
    clean_df['high'] = df[actual_columns['high']].astype(float)
    clean_df['low'] = df[actual_columns['low']].astype(float)
    clean_df['close'] = df[actual_columns['close']].astype(float)
    
    if 'volume' in actual_columns:
        clean_df['volume'] = df[actual_columns['volume']].astype(float)
    else:
        clean_df['volume'] = 1.0  # Default volume
    
    # Add datetime if available
    if 'datetime' in df.columns:
        clean_df['datetime'] = df['datetime']
    
    # Remove invalid data
    initial_count = len(clean_df)
    clean_df = clean_df.dropna()
    clean_df = clean_df[(clean_df['high'] >= clean_df['low']) & 
                       (clean_df['high'] >= clean_df['open']) & 
                       (clean_df['high'] >= clean_df['close']) &
                       (clean_df['low'] <= clean_df['open']) & 
                       (clean_df['low'] <= clean_df['close'])]
    
    final_count = len(clean_df)
    logger.info(f"📊 Cleaned data: {initial_count:,} → {final_count:,} records")
    
    if final_count < 1000:
        raise Exception("❌ Insufficient clean data for training")
    
    # Generate unified features
    logger.info("🔧 Generating unified features (19 features)...")
    
    try:
        # Create market data objects for feature engine
        market_data_list = []
        for idx, row in clean_df.iterrows():
            market_data = {
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            market_data_list.append(market_data)
        
        # Generate features in batches to manage memory
        batch_size = 10000
        all_features = []
        
        for i in range(0, len(market_data_list), batch_size):
            batch = market_data_list[i:i+batch_size]
            batch_features = []
            
            for market_data in batch:
                features = feature_engine.get_unified_features(market_data)
                batch_features.append(features.flatten())
            
            all_features.extend(batch_features)
            
            if i % (batch_size * 5) == 0:
                logger.info(f"  Processed {i:,}/{len(market_data_list):,} records...")
        
        X = np.array(all_features)
        logger.info(f"✅ Features shape: {X.shape}")
        
    except Exception as e:
        logger.error(f"❌ Error generating features: {e}")
        # Fallback to basic features
        logger.info("🔄 Using fallback basic features...")
        X = np.column_stack([
            clean_df['open'].values,
            clean_df['high'].values,
            clean_df['low'].values,
            clean_df['close'].values,
            clean_df['volume'].values
        ])
    
    # Generate targets (next price direction)
    logger.info("🎯 Generating training targets...")
    
    close_prices = clean_df['close'].values
    price_changes = np.diff(close_prices)
    
    # Binary classification: 1 if price goes up, 0 if down
    y = (price_changes > 0).astype(int)
    
    # Remove last row from X to match y length
    X = X[:-1]
    
    logger.info(f"✅ Training data prepared:")
    logger.info(f"  Features (X): {X.shape}")
    logger.info(f"  Targets (y): {y.shape}")
    logger.info(f"  Target distribution: {np.bincount(y)}")
    
    return X, y

def run_comprehensive_training():
    """Run comprehensive training with all data"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 STARTING COMPREHENSIVE TRAINING")
        logger.info("=" * 60)
        
        # Initialize system
        config = SystemConfig()
        config.enable_integrated_training = True
        system = UltimateXAUSystem(config)
        
        if not system.training_system:
            raise Exception("❌ Training system not initialized")
        
        # Load all data
        logger.info("📊 Phase 1: Data Loading")
        combined_data = load_all_data()
        
        # Prepare training data
        logger.info("🔧 Phase 2: Data Preparation")
        X, y = prepare_comprehensive_training_data(combined_data)
        
        # Add training data to system
        logger.info("📝 Phase 3: Adding Data to Training System")
        
        # Simulate real market data collection
        for i in range(min(len(X), 50000)):  # Limit to 50K for memory management
            market_data = {
                'features': X[i],
                'target': y[i],
                'timestamp': datetime.now()
            }
            system.training_system.collect_training_data(market_data)
            
            if i % 10000 == 0:
                logger.info(f"  Added {i:,}/{min(len(X), 50000):,} data points...")
        
        # Start training
        logger.info("🤖 Phase 4: Model Training")
        logger.info("Training 4 unified models: LSTM, CNN, Hybrid, Dense")
        
        training_result = system.start_training()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results/comprehensive_training_{timestamp}.json"
        
        os.makedirs("training_results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(training_result, f, indent=2, default=str)
        
        logger.info("✅ COMPREHENSIVE TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"📊 Results saved: {results_file}")
        
        # Print summary
        if training_result.get('status') == 'completed':
            logger.info("🎯 TRAINING SUMMARY:")
            logger.info(f"  Models trained: {training_result.get('models_trained', 0)}")
            logger.info(f"  Training time: {training_result.get('training_time', 'N/A')}")
            
            if 'results' in training_result:
                for model_name, result in training_result['results'].items():
                    if 'error' not in result:
                        acc = result.get('val_accuracy', 0)
                        logger.info(f"  {model_name.upper()}: {acc:.4f} accuracy")
        
        return training_result
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("🚀 AI3.0 COMPREHENSIVE TRAINING")
    print("Training ALL available data with Unified System")
    print("=" * 60)
    
    result = run_comprehensive_training()
    
    if result.get('status') == 'completed':
        print("\n✅ SUCCESS: Comprehensive training completed!")
        print(f"🤖 Models trained: {result.get('models_trained', 0)}")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}") 