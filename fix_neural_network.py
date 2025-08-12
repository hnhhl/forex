"""
Script để sửa NeuralNetworkSystem process method
"""

def fix_neural_network():
    file_path = "src/core/ultimate_xau_system.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sửa method process của NeuralNetworkSystem
    broken_method = """    def process(self, data: pd.DataFrame) -> Dict:
        try:
            predictions = {}
            
            # Process with each model
            for model_name, model in self.models.items():
                prediction = max(0.1, min(0.9, abs(prediction) / 100.0 if abs(prediction) > 1 else prediction))}
            
        except Exception as e:
            self.log_error(e)
            return {'error': str(e)}"""
    
    fixed_method = """    def process(self, data: pd.DataFrame) -> Dict:
        try:
            predictions = {}
            
            # Process with each model
            for model_name, model in self.models.items():
                try:
                    prediction_result = self._predict_with_model(model_name, model, data)
                    predictions[model_name] = prediction_result
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = {'prediction': 0.5, 'confidence': 0.3}
            
            # Ensemble prediction
            if predictions:
                ensemble_result = self._ensemble_predict(predictions)
                prediction = ensemble_result.get('prediction', 0.5)
                confidence = ensemble_result.get('confidence', 0.5)
            else:
                prediction = 0.5
                confidence = 0.3
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'individual_predictions': predictions,
                'ensemble_prediction': {
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'method': 'neural_ensemble'
                },
                'model_count': len(self.models),
                'active_models': len(predictions)
            }
            
        except Exception as e:
            self.log_error(e)
            return {
                'prediction': 0.5,
                'confidence': 0.3,
                'error': str(e)
            }"""
    
    content = content.replace(broken_method, fixed_method)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed NeuralNetworkSystem process method!")

if __name__ == "__main__":
    fix_neural_network() 