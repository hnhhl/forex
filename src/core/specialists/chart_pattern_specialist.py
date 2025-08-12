"""
Chart Pattern Specialist
================================================================================
Technical Specialist chuyên về Chart Pattern Recognition
Thuộc Technical Category trong Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import stats
from scipy.signal import find_peaks

from .base_specialist import BaseSpecialist, SpecialistVote

logger = logging.getLogger(__name__)


class ChartPatternSpecialist(BaseSpecialist):
    """Chart Pattern Specialist - Chuyên gia phân tích chart patterns"""
    
    def __init__(self, min_pattern_length: int = 20, confidence_threshold: float = 0.6):
        super().__init__(
            name="Chart_Pattern_Specialist",
            category="Pattern",
            description=f"Chart pattern analysis với min_length={min_pattern_length}"
        )
        
        self.min_pattern_length = min_pattern_length
        self.confidence_threshold = confidence_threshold
        self.min_data_points = max(30, min_pattern_length)
        
        logger.info(f"Chart Pattern Specialist initialized")
    
    def detect_triangular_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect triangular patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        if len(data) < self.min_pattern_length:
            return patterns
        
        try:
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            # Find significant peaks and troughs
            high_peaks, _ = find_peaks(high_prices, distance=5, prominence=np.std(high_prices)*0.5)
            low_peaks, _ = find_peaks(-low_prices, distance=5, prominence=np.std(low_prices)*0.5)
            
            # Detect ascending triangle
            ascending = self._detect_ascending_triangle(data, high_peaks, low_peaks)
            if ascending:
                patterns.append(ascending)
            
            # Detect descending triangle
            descending = self._detect_descending_triangle(data, high_peaks, low_peaks)
            if descending:
                patterns.append(descending)
            
            # Detect symmetrical triangle
            symmetrical = self._detect_symmetrical_triangle(data, high_peaks, low_peaks)
            if symmetrical:
                patterns.append(symmetrical)
        
        except Exception as e:
            self.logger.error(f"Error detecting triangular patterns: {e}")
        
        return patterns
    
    def _detect_ascending_triangle(self, data: pd.DataFrame, high_peaks: np.ndarray, low_peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ascending triangle pattern"""
        
        if len(high_peaks) < 2 or len(low_peaks) < 2:
            return None
        
        try:
            # Get recent peaks
            recent_highs = high_peaks[-3:] if len(high_peaks) >= 3 else high_peaks
            recent_lows = low_peaks[-3:] if len(low_peaks) >= 3 else low_peaks
            
            # Check for horizontal resistance (similar highs)
            high_values = data['high'].iloc[recent_highs].values
            high_similarity = np.std(high_values) / np.mean(high_values) < 0.02
            
            # Check for ascending support (rising lows)
            low_values = data['low'].iloc[recent_lows].values
            if len(low_values) >= 2:
                slope, _, r_value, _, _ = stats.linregress(recent_lows, low_values)
                ascending_support = slope > 0 and r_value > 0.7
            else:
                ascending_support = False
            
            if high_similarity and ascending_support:
                resistance_level = np.mean(high_values)
                
                return {
                    'pattern_name': 'Ascending Triangle',
                    'pattern_type': 'BULLISH',
                    'confidence': min(0.9, 0.6 + abs(r_value) * 0.3),
                    'resistance_level': resistance_level,
                    'support_slope': slope,
                    'target_price': resistance_level * 1.05,
                    'stop_loss': low_values[-1] * 0.98
                }
        
        except Exception as e:
            self.logger.error(f"Error in ascending triangle detection: {e}")
        
        return None
    
    def _detect_descending_triangle(self, data: pd.DataFrame, high_peaks: np.ndarray, low_peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect descending triangle pattern"""
        
        if len(high_peaks) < 2 or len(low_peaks) < 2:
            return None
        
        try:
            # Get recent peaks
            recent_highs = high_peaks[-3:] if len(high_peaks) >= 3 else high_peaks
            recent_lows = low_peaks[-3:] if len(low_peaks) >= 3 else low_peaks
            
            # Check for horizontal support (similar lows)
            low_values = data['low'].iloc[recent_lows].values
            low_similarity = np.std(low_values) / np.mean(low_values) < 0.02
            
            # Check for descending resistance (falling highs)
            high_values = data['high'].iloc[recent_highs].values
            if len(high_values) >= 2:
                slope, _, r_value, _, _ = stats.linregress(recent_highs, high_values)
                descending_resistance = slope < 0 and r_value < -0.7
            else:
                descending_resistance = False
            
            if low_similarity and descending_resistance:
                support_level = np.mean(low_values)
                
                return {
                    'pattern_name': 'Descending Triangle',
                    'pattern_type': 'BEARISH',
                    'confidence': min(0.9, 0.6 + abs(r_value) * 0.3),
                    'support_level': support_level,
                    'resistance_slope': slope,
                    'target_price': support_level * 0.95,
                    'stop_loss': high_values[-1] * 1.02
                }
        
        except Exception as e:
            self.logger.error(f"Error in descending triangle detection: {e}")
        
        return None
    
    def _detect_symmetrical_triangle(self, data: pd.DataFrame, high_peaks: np.ndarray, low_peaks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect symmetrical triangle pattern"""
        
        if len(high_peaks) < 3 or len(low_peaks) < 3:
            return None
        
        try:
            # Take recent peaks
            recent_highs = high_peaks[-3:]
            recent_lows = low_peaks[-3:]
            
            # Get values
            high_values = data['high'].iloc[recent_highs].values
            low_values = data['low'].iloc[recent_lows].values
            
            # Check for converging trendlines
            high_slope, _, high_r, _, _ = stats.linregress(recent_highs, high_values)
            low_slope, _, low_r, _, _ = stats.linregress(recent_lows, low_values)
            
            # Symmetrical triangle: descending highs and ascending lows
            converging = (high_slope < 0 and low_slope > 0 and 
                         abs(high_r) > 0.7 and abs(low_r) > 0.7)
            
            if converging:
                # Calculate convergence point
                convergence_price = (high_values[0] + low_values[0]) / 2
                
                return {
                    'pattern_name': 'Symmetrical Triangle',
                    'pattern_type': 'NEUTRAL',
                    'confidence': min(0.85, 0.5 + (abs(high_r) + abs(low_r)) * 0.175),
                    'high_slope': high_slope,
                    'low_slope': low_slope,
                    'convergence_price': convergence_price,
                    'target_price': convergence_price
                }
        
        except Exception as e:
            self.logger.error(f"Error in symmetrical triangle detection: {e}")
        
        return None
    
    def detect_head_and_shoulders(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect Head and Shoulders pattern"""
        
        if len(data) < 30:
            return None
        
        try:
            highs = data['high'].rolling(window=3).max()
            
            # Look for three peaks
            peaks = []
            for i in range(10, len(data) - 10):
                if highs.iloc[i] == data['high'].iloc[i-2:i+3].max():
                    peaks.append((i, data['high'].iloc[i]))
            
            if len(peaks) >= 3:
                # Sort by height
                peaks.sort(key=lambda x: x[1], reverse=True)
                
                # Check H&S pattern
                head = peaks[0]
                left_shoulder = None
                right_shoulder = None
                
                for peak in peaks[1:]:
                    if peak[0] < head[0] and left_shoulder is None:
                        left_shoulder = peak
                    elif peak[0] > head[0] and right_shoulder is None:
                        right_shoulder = peak
                
                if left_shoulder and right_shoulder:
                    # Validate pattern
                    if (abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05 and
                        head[1] > left_shoulder[1] * 1.05):
                        
                        return {
                            'pattern_name': 'Head and Shoulders',
                            'pattern_type': 'BEARISH',
                            'confidence': 0.8,
                            'head_price': head[1],
                            'left_shoulder': left_shoulder[1],
                            'right_shoulder': right_shoulder[1],
                            'target_price': min(left_shoulder[1], right_shoulder[1]) * 0.95,
                            'stop_loss': head[1] * 1.02
                        }
        
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return None
    
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """Analyze chart patterns và generate vote"""
        
        if not self.enabled:
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Specialist is disabled",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        if not self.validate_data(data):
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Invalid or insufficient data",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        try:
            # Detect all pattern types
            all_patterns = []
            
            # Triangular patterns
            triangular_patterns = self.detect_triangular_patterns(data)
            all_patterns.extend(triangular_patterns)
            
            # Head and shoulders
            head_shoulders = self.detect_head_and_shoulders(data)
            if head_shoulders:
                all_patterns.append(head_shoulders)
            
            # Analyze patterns for voting
            vote = "HOLD"
            confidence = 0.0
            best_pattern = None
            
            if all_patterns:
                # Find highest confidence pattern
                best_pattern = max(all_patterns, key=lambda p: p['confidence'])
                
                if best_pattern['confidence'] >= self.confidence_threshold:
                    if best_pattern['pattern_type'] == 'BULLISH':
                        vote = "BUY"
                    elif best_pattern['pattern_type'] == 'BEARISH':
                        vote = "SELL"
                    else:
                        vote = "HOLD"
                    
                    confidence = best_pattern['confidence']
            
            analysis_result = {
                'current_price': current_price,
                'patterns_detected': len(all_patterns),
                'best_pattern': best_pattern,
                'all_patterns': all_patterns
            }
            
            final_confidence = self.calculate_confidence(analysis_result)
            reasoning = self.generate_reasoning(analysis_result, vote)
            
            specialist_vote = SpecialistVote(
                specialist_name=self.name,
                vote=vote,
                confidence=final_confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                technical_data=analysis_result
            )
            
            self.vote_history.append({
                'vote': vote,
                'confidence': final_confidence,
                'patterns_count': len(all_patterns),
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Chart Pattern Analysis: {len(all_patterns)} patterns, Vote={vote}, Confidence={final_confidence:.2f}")
            
            return specialist_vote
            
        except Exception as e:
            self.logger.error(f"Error in chart pattern analysis: {e}")
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                timestamp=datetime.now(),
                technical_data={}
            )
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        base_confidence = 0.0
        
        if analysis_result['best_pattern']:
            base_confidence = analysis_result['best_pattern']['confidence']
        
        # Adjust for number of patterns detected
        patterns_count = analysis_result['patterns_detected']
        if patterns_count > 1:
            base_confidence *= min(1.2, 1.0 + patterns_count * 0.1)
        
        # Adjust for recent accuracy
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy > 0.6:
            base_confidence *= 1.1
        elif recent_accuracy < 0.4:
            base_confidence *= 0.8
        
        return min(1.0, max(0.0, base_confidence))
    
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """Generate reasoning"""
        patterns_count = analysis_result['patterns_detected']
        best_pattern = analysis_result['best_pattern']
        
        if patterns_count == 0:
            return "No significant chart patterns detected. Suggesting HOLD"
        
        reasoning = f"Detected {patterns_count} chart pattern(s)"
        
        if best_pattern:
            reasoning += f". Best pattern: {best_pattern['pattern_name']} ({best_pattern['pattern_type']}) with {best_pattern['confidence']:.2f} confidence"
        
        vote_reasoning = {
            "BUY": "Suggesting BUY - bullish chart patterns detected",
            "SELL": "Suggesting SELL - bearish chart patterns detected", 
            "HOLD": "Suggesting HOLD - neutral or mixed chart patterns"
        }
        
        return f"{reasoning}. {vote_reasoning.get(vote, '')}"


def create_chart_pattern_specialist(min_length: int = 20, threshold: float = 0.6) -> ChartPatternSpecialist:
    """Factory function to create Chart Pattern Specialist"""
    return ChartPatternSpecialist(min_pattern_length=min_length, confidence_threshold=threshold) 