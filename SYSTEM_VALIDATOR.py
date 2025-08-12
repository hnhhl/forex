#!/usr/bin/env python3
"""
AI3.0 SYSTEM VALIDATOR
Kiểm tra tính toàn vẹn và hoạt động của hệ thống
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, 'src')

class SystemValidator:
    """Validator cho AI3.0 system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    def validate_complete_system(self) -> Dict[str, any]:
        """Validate toàn bộ hệ thống"""
        print("🔍 AI3.0 SYSTEM VALIDATION")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'components': {},
            'scores': {},
            'recommendations': []
        }
        
        # 1. Validate file structure
        print("\n📁 Validating file structure...")
        results['components']['file_structure'] = self._validate_file_structure()
        
        # 2. Validate imports
        print("\n📦 Validating imports...")
        results['components']['imports'] = self._validate_imports()
        
        # 3. Validate AI models
        print("\n🤖 Validating AI models...")
        results['components']['ai_models'] = self._validate_ai_models()
        
        # 4. Validate specialists
        print("\n👥 Validating specialists...")
        results['components']['specialists'] = self._validate_specialists()
        
        # Calculate overall score
        results['scores'] = self._calculate_scores(results['components'])
        results['overall_status'] = 'HEALTHY' if results['scores']['overall'] >= 80 else 'NEEDS_ATTENTION'
        results['recommendations'] = self._generate_recommendations(results['components'])
        
        # Display results
        self._display_validation_results(results)
        
        return results
    
    def _validate_file_structure(self) -> Dict[str, any]:
        """Validate file structure"""
        required_files = [
            'UNIFIED_AI3_MASTER_SYSTEM.py',
            'SYSTEM_LAUNCHER.py',
            'config/unified_system_config.json'
        ]
        
        result = {
            'status': 'PASS',
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check required files
        required_score = 0
        for file in required_files:
            exists = os.path.exists(file)
            result['details'][file] = 'FOUND' if exists else 'MISSING'
            if exists:
                required_score += 1
            else:
                result['issues'].append(f"Missing required file: {file}")
        
        # Calculate score
        result['score'] = int((required_score / len(required_files)) * 100)
        
        if result['score'] < 70:
            result['status'] = 'FAIL'
        elif result['score'] < 90:
            result['status'] = 'WARNING'
        
        print(f"   File structure: {result['status']} ({result['score']}%)")
        return result
    
    def _validate_imports(self) -> Dict[str, any]:
        """Validate critical imports"""
        critical_imports = [
            'numpy',
            'pandas',
            'asyncio',
            'datetime',
            'json'
        ]
        
        result = {
            'status': 'PASS',
            'score': 0,
            'details': {},
            'issues': []
        }
        
        successful_imports = 0
        for module_name in critical_imports:
            try:
                __import__(module_name)
                result['details'][module_name] = 'SUCCESS'
                successful_imports += 1
            except ImportError as e:
                result['details'][module_name] = f'FAILED: {str(e)}'
                result['issues'].append(f"Import failed: {module_name}")
        
        result['score'] = int((successful_imports / len(critical_imports)) * 100)
        
        if result['score'] < 60:
            result['status'] = 'FAIL'
        elif result['score'] < 80:
            result['status'] = 'WARNING'
        
        print(f"   Imports: {result['status']} ({successful_imports}/{len(critical_imports)})")
        return result
    
    def _validate_ai_models(self) -> Dict[str, any]:
        """Validate AI models"""
        model_files = [
            'trained_models_optimized/neural_network_H1.keras',
            'trained_models_optimized/neural_network_H4.keras',
            'trained_models_optimized/neural_network_D1.keras'
        ]
        
        result = {
            'status': 'PASS',
            'score': 0,
            'details': {},
            'issues': []
        }
        
        available_models = 0
        for model_path in model_files:
            if os.path.exists(model_path):
                result['details'][model_path] = 'FOUND'
                available_models += 1
            else:
                result['details'][model_path] = 'MISSING'
                result['issues'].append(f"Missing model: {model_path}")
        
        result['score'] = int((available_models / len(model_files)) * 100)
        
        if result['score'] == 0:
            result['status'] = 'FAIL'
        elif result['score'] < 50:
            result['status'] = 'WARNING'
        
        print(f"   AI Models: {result['status']} ({available_models}/{len(model_files)})")
        return result
    
    def _validate_specialists(self) -> Dict[str, any]:
        """Validate specialists"""
        result = {
            'status': 'PASS',
            'score': 0,
            'details': {},
            'issues': []
        }
        
        specialist_files = [
            'src/core/specialists/__init__.py',
            'src/core/specialists/democratic_voting_engine.py',
            'src/core/specialists/base_specialist.py'
        ]
        
        available_files = 0
        for file_path in specialist_files:
            if os.path.exists(file_path):
                result['details'][file_path] = 'FOUND'
                available_files += 1
            else:
                result['details'][file_path] = 'MISSING'
                result['issues'].append(f"Missing specialist file: {file_path}")
        
        result['score'] = int((available_files / len(specialist_files)) * 100)
        
        if result['score'] < 70:
            result['status'] = 'FAIL'
        elif result['score'] < 90:
            result['status'] = 'WARNING'
        
        print(f"   Specialists: {result['status']} ({available_files}/{len(specialist_files)})")
        return result
    
    def _calculate_scores(self, components: Dict) -> Dict[str, int]:
        """Calculate overall scores"""
        scores = {
            'file_structure': components.get('file_structure', {}).get('score', 0),
            'imports': components.get('imports', {}).get('score', 0),
            'ai_models': components.get('ai_models', {}).get('score', 0),
            'specialists': components.get('specialists', {}).get('score', 0)
        }
        
        # Simple average for overall score
        overall = sum(scores.values()) / len(scores)
        scores['overall'] = int(overall)
        
        return scores
    
    def _generate_recommendations(self, components: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for component_name, component_data in components.items():
            if component_data.get('status') == 'FAIL':
                recommendations.append(f"🔴 CRITICAL: Fix {component_name}")
            elif component_data.get('status') == 'WARNING':
                recommendations.append(f"🟡 WARNING: Improve {component_name}")
        
        if not recommendations:
            recommendations.append("✅ System is healthy")
        
        return recommendations
    
    def _display_validation_results(self, results: Dict):
        """Display validation results"""
        print(f"\n📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Status: {results['overall_status']}")
        print(f"Overall Score: {results['scores']['overall']}%")
        
        print(f"\n📊 Component Scores:")
        for component, score in results['scores'].items():
            if component != 'overall':
                status_icon = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                print(f"   {status_icon} {component}: {score}%")
        
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   {rec}")

def main():
    """Main validation function"""
    validator = SystemValidator()
    results = validator.validate_complete_system()
    return results

if __name__ == "__main__":
    main() 