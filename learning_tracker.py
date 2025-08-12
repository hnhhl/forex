# -*- coding: utf-8 -*-
"""Learning Tracker - Quick Win #3: Real-time Learning System"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional

class LearningTracker:
    """Track predictions and outcomes for continuous learning"""
    
    def __init__(self, data_file: str = "learning_data/predictions_log.csv"):
        self.data_file = data_file
        self.ensure_data_directory()
        self.load_existing_data()
    
    def ensure_data_directory(self):
        """Ensure learning_data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def load_existing_data(self):
        """Load existing prediction data"""
        try:
            if os.path.exists(self.data_file):
                self.predictions_df = pd.read_csv(self.data_file)
                print(f"✅ Loaded {len(self.predictions_df)} existing predictions")
            else:
                self.predictions_df = pd.DataFrame(columns=[
                    'timestamp', 'prediction', 'confidence', 'actual_outcome',
                    'market_price', 'signal', 'system_votes', 'accuracy'
                ])
                print("📊 Created new predictions log")
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
            self.predictions_df = pd.DataFrame(columns=[
                'timestamp', 'prediction', 'confidence', 'actual_outcome',
                'market_price', 'signal', 'system_votes', 'accuracy'
            ])
    
    def save_prediction(self, prediction: float, confidence: float, 
                       signal: str, market_price: float, 
                       system_votes: Dict, timestamp: Optional[datetime] = None):
        """Save a new prediction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        new_prediction = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'actual_outcome': None,  # Will be updated later
            'market_price': market_price,
            'signal': signal,
            'system_votes': json.dumps(system_votes),
            'accuracy': None  # Will be calculated when outcome is known
        }
        
        # Add to dataframe
        self.predictions_df = pd.concat([
            self.predictions_df,
            pd.DataFrame([new_prediction])
        ], ignore_index=True)
        
        # Save to file
        self.save_to_file()
        
        print(f"💾 Saved prediction: {signal} ({confidence:.1f}% confidence)")
    
    def update_outcome(self, timestamp: datetime, actual_outcome: float):
        """Update prediction with actual outcome"""
        try:
            # Find prediction by timestamp
            mask = pd.to_datetime(self.predictions_df['timestamp']) == timestamp
            
            if mask.any():
                idx = self.predictions_df[mask].index[0]
                self.predictions_df.loc[idx, 'actual_outcome'] = actual_outcome
                
                # Calculate accuracy
                prediction = self.predictions_df.loc[idx, 'prediction']
                accuracy = 1.0 - abs(prediction - actual_outcome)
                self.predictions_df.loc[idx, 'accuracy'] = accuracy
                
                self.save_to_file()
                print(f"✅ Updated outcome: accuracy = {accuracy:.3f}")
            else:
                print(f"⚠️ Prediction not found for timestamp: {timestamp}")
                
        except Exception as e:
            print(f"❌ Error updating outcome: {e}")
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """Get performance metrics for recent period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = self.predictions_df[
                pd.to_datetime(self.predictions_df['timestamp']) >= cutoff_date
            ]
            
            if len(recent_data) == 0:
                return {'error': 'No recent data available'}
            
            # Calculate metrics
            completed_predictions = recent_data[recent_data['accuracy'].notna()]
            
            if len(completed_predictions) == 0:
                return {'error': 'No completed predictions in recent period'}
            
            metrics = {
                'total_predictions': len(recent_data),
                'completed_predictions': len(completed_predictions),
                'average_accuracy': completed_predictions['accuracy'].mean(),
                'average_confidence': recent_data['confidence'].mean(),
                'signal_distribution': recent_data['signal'].value_counts().to_dict(),
                'best_accuracy': completed_predictions['accuracy'].max(),
                'worst_accuracy': completed_predictions['accuracy'].min(),
                'period_days': days
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating recent performance: {e}")
            return {'error': str(e)}
    
    def get_system_performance(self) -> Dict:
        """Get performance by system votes"""
        try:
            completed_data = self.predictions_df[self.predictions_df['accuracy'].notna()]
            
            if len(completed_data) == 0:
                return {'error': 'No completed predictions available'}
            
            system_performance = {}
            
            for _, row in completed_data.iterrows():
                try:
                    votes = json.loads(row['system_votes'])
                    accuracy = row['accuracy']
                    
                    for system, vote in votes.items():
                        if system not in system_performance:
                            system_performance[system] = {
                                'votes': [],
                                'accuracies': [],
                                'total_votes': 0
                            }
                        
                        system_performance[system]['votes'].append(vote)
                        system_performance[system]['accuracies'].append(accuracy)
                        system_performance[system]['total_votes'] += 1
                
                except json.JSONDecodeError:
                    continue
            
            # Calculate averages
            for system in system_performance:
                perf = system_performance[system]
                perf['average_accuracy'] = np.mean(perf['accuracies'])
                perf['vote_distribution'] = pd.Series(perf['votes']).value_counts().to_dict()
                del perf['votes']  # Remove raw data
                del perf['accuracies']  # Remove raw data
            
            return system_performance
            
        except Exception as e:
            print(f"❌ Error calculating system performance: {e}")
            return {'error': str(e)}
    
    def save_to_file(self):
        """Save predictions to CSV file"""
        try:
            self.predictions_df.to_csv(self.data_file, index=False)
        except Exception as e:
            print(f"❌ Error saving to file: {e}")
    
    def get_learning_insights(self) -> Dict:
        """Get insights for model improvement"""
        try:
            completed_data = self.predictions_df[self.predictions_df['accuracy'].notna()]
            
            if len(completed_data) == 0:
                return {'error': 'No completed predictions available'}
            
            insights = {
                'confidence_vs_accuracy': self._analyze_confidence_accuracy(completed_data),
                'signal_performance': self._analyze_signal_performance(completed_data),
                'time_patterns': self._analyze_time_patterns(completed_data),
                'improvement_suggestions': self._generate_suggestions(completed_data)
            }
            
            return insights
            
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
            return {'error': str(e)}
    
    def _analyze_confidence_accuracy(self, data: pd.DataFrame) -> Dict:
        """Analyze relationship between confidence and accuracy"""
        # Bin confidence levels
        data['confidence_bin'] = pd.cut(data['confidence'], bins=[0, 30, 50, 70, 100], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
        
        confidence_analysis = data.groupby('confidence_bin')['accuracy'].agg([
            'mean', 'count', 'std'
        ]).round(3).to_dict()
        
        return confidence_analysis
    
    def _analyze_signal_performance(self, data: pd.DataFrame) -> Dict:
        """Analyze performance by signal type"""
        signal_analysis = data.groupby('signal')['accuracy'].agg([
            'mean', 'count', 'std'
        ]).round(3).to_dict()
        
        return signal_analysis
    
    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze performance patterns over time"""
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        
        time_analysis = data.groupby('hour')['accuracy'].agg([
            'mean', 'count'
        ]).round(3).to_dict()
        
        return time_analysis
    
    def _generate_suggestions(self, data: pd.DataFrame) -> List[str]:
        """Generate improvement suggestions based on data"""
        suggestions = []
        
        # Low accuracy suggestions
        avg_accuracy = data['accuracy'].mean()
        if avg_accuracy < 0.6:
            suggestions.append("Consider retraining models - average accuracy is below 60%")
        
        # Confidence calibration
        high_conf_low_acc = data[(data['confidence'] > 70) & (data['accuracy'] < 0.5)]
        if len(high_conf_low_acc) > len(data) * 0.1:
            suggestions.append("Confidence calibration needed - high confidence with low accuracy")
        
        # Signal distribution
        signal_counts = data['signal'].value_counts()
        if 'HOLD' in signal_counts and signal_counts['HOLD'] > len(data) * 0.8:
            suggestions.append("Too many HOLD signals - consider adjusting thresholds")
        
        return suggestions

# Performance Monitor
class PerformanceMonitor:
    """Monitor system performance in real-time"""
    
    def __init__(self, tracker: LearningTracker):
        self.tracker = tracker
    
    def track_confidence_trend(self, days: int = 30) -> Dict:
        """Track confidence trend over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = self.tracker.predictions_df[
                pd.to_datetime(self.tracker.predictions_df['timestamp']) >= cutoff_date
            ]
            
            if len(recent_data) == 0:
                return {'error': 'No recent data'}
            
            # Group by day and calculate average confidence
            recent_data['date'] = pd.to_datetime(recent_data['timestamp']).dt.date
            daily_confidence = recent_data.groupby('date')['confidence'].mean()
            
            trend = {
                'dates': [str(date) for date in daily_confidence.index],
                'confidence_values': daily_confidence.tolist(),
                'trend_direction': 'up' if daily_confidence.iloc[-1] > daily_confidence.iloc[0] else 'down',
                'average_confidence': daily_confidence.mean()
            }
            
            return trend
            
        except Exception as e:
            return {'error': str(e)}
    
    def track_accuracy_trend(self, days: int = 30) -> Dict:
        """Track accuracy trend over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_data = self.tracker.predictions_df[
                (pd.to_datetime(self.tracker.predictions_df['timestamp']) >= cutoff_date) &
                (self.tracker.predictions_df['accuracy'].notna())
            ]
            
            if len(recent_data) == 0:
                return {'error': 'No recent completed predictions'}
            
            # Group by day and calculate average accuracy
            recent_data['date'] = pd.to_datetime(recent_data['timestamp']).dt.date
            daily_accuracy = recent_data.groupby('date')['accuracy'].mean()
            
            trend = {
                'dates': [str(date) for date in daily_accuracy.index],
                'accuracy_values': daily_accuracy.tolist(),
                'trend_direction': 'up' if daily_accuracy.iloc[-1] > daily_accuracy.iloc[0] else 'down',
                'average_accuracy': daily_accuracy.mean()
            }
            
            return trend
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_performance_drop(self, threshold: float = 0.1) -> Dict:
        """Detect significant performance drops"""
        try:
            recent_7_days = self.tracker.get_recent_performance(7)
            recent_30_days = self.tracker.get_recent_performance(30)
            
            if 'error' in recent_7_days or 'error' in recent_30_days:
                return {'error': 'Insufficient data for comparison'}
            
            accuracy_drop = recent_30_days['average_accuracy'] - recent_7_days['average_accuracy']
            confidence_drop = recent_30_days['average_confidence'] - recent_7_days['average_confidence']
            
            alert = {
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop,
                'alert_triggered': accuracy_drop > threshold or confidence_drop > threshold * 100,
                'severity': 'high' if accuracy_drop > threshold * 2 else 'medium' if accuracy_drop > threshold else 'low'
            }
            
            return alert
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    print("🎯 QUICK WIN #3: REAL-TIME LEARNING SYSTEM")
    print("="*60)
    
    # Test the learning tracker
    tracker = LearningTracker()
    monitor = PerformanceMonitor(tracker)
    
    print("✅ Learning Tracker initialized")
    print("✅ Performance Monitor initialized")
    print("🔄 Ready for real-time learning!")
    
    # Example usage
    print("\n📊 Example usage:")
    print("tracker.save_prediction(0.7, 85.0, 'BUY', 3350.0, {'Neural': 'BUY', 'AI2': 'HOLD'})")
    print("tracker.update_outcome(timestamp, 0.75)  # Actual outcome")
    print("performance = tracker.get_recent_performance(7)")
    print("insights = tracker.get_learning_insights()") 