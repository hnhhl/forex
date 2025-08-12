
def prepare_5_features_with_volume(data):
    """
    Enhanced data preparation ensuring 5 features with volume
    Compatible với AI3.0 neural models
    """
    try:
        # Required 5 features
        required_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Ensure volume column exists
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'real_volume' in df.columns:
                df['volume'] = df['real_volume']
            else:
                # Create synthetic volume from price movement
                df['volume'] = np.abs(df['close'] - df['open']) * 1000
        
        # Validate all 5 features exist
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and validate data
        feature_data = df[required_features]
        
        if len(feature_data) < 60:
            raise ValueError(f"Insufficient data: {len(feature_data)} < 60")
        
        # Prepare for neural models
        features_array = feature_data.tail(60).values
        features_reshaped = features_array.reshape(1, 60, 5)
        
        return {
            'success': True,
            'features': features_reshaped,
            'shape': features_reshaped.shape,
            'columns': required_features,
            'message': f'Successfully prepared 5 features: {required_features}'
        }
        
    except Exception as e:
        return {
            'success': False,
            'features': None,
            'shape': None,
            'columns': None,
            'message': f'Error: {str(e)}'
        }
