# USAGE EXAMPLE - How to use in your trading system

from production_model_loader import production_model_loader

# Load best model and make prediction
def get_trading_signal(features):
    """Get trading signal using best available model"""
    
    # Reshape features for prediction
    X = features.reshape(1, -1)
    
    # Get prediction from best model
    result = production_model_loader.predict_with_best_model(X)
    
    # Convert to trading signal
    prediction = result['prediction']
    confidence = result['confidence']
    model_used = result['model_used']
    
    # Decision logic
    if prediction > 0.6 and confidence > 0.7:
        return {
            'signal': 'BUY',
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }
    elif prediction < 0.4 and confidence > 0.7:
        return {
            'signal': 'SELL', 
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }
    else:
        return {
            'signal': 'HOLD',
            'confidence': confidence,
            'model': model_used,
            'prediction_value': prediction
        }

# Example usage
# signal = get_trading_signal(your_features)
# print(f"Signal: {signal['signal']}, Confidence: {signal['confidence']:.2f}")
