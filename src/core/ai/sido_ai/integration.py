"""
SIDO AI Integration Layer
Connects all SIDO AI modules with main system
"""

from typing import Dict, List, Any
import importlib
import sys
import os

class SIDOAIIntegration:
    def __init__(self):
        self.modules = {}
        self.active_modules = []
        self.integration_status = "ready"
        
    def initialize_modules(self):
        """Initialize all SIDO AI modules"""
        module_list = [
            'pattern_recognition',
            'market_analysis', 
            'signal_generation',
            'risk_assessment',
            'portfolio_optimization'
        ]
        
        for module_name in module_list:
            try:
                module = importlib.import_module(f'core.ai.sido_ai.modules.{module_name}')
                self.modules[module_name] = module
                self.active_modules.append(module_name)
            except ImportError:
                print(f"Warning: Could not import {module_name}")
                
        return len(self.active_modules)
        
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through all modules"""
        results = {}
        
        for module_name in self.active_modules:
            if module_name in self.modules:
                try:
                    result = self.modules[module_name].process(market_data)
                    results[module_name] = result
                except Exception as e:
                    results[module_name] = {'error': str(e)}
                    
        return {
            'sido_ai_results': results,
            'active_modules': len(self.active_modules),
            'success_rate': len([r for r in results.values() if 'error' not in r]) / len(results)
        }
