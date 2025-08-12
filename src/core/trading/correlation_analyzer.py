"""
Correlation Analyzer Module
Advanced correlation analysis for portfolio management và risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"
    DYNAMIC = "dynamic"


class ClusteringMethod(Enum):
    """Clustering methods for correlation analysis"""
    HIERARCHICAL = "hierarchical"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"


@dataclass
class CorrelationMetrics:
    """Correlation analysis metrics"""
    correlation_matrix: np.ndarray
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_stability: float
    diversification_ratio: float
    effective_assets: float
    concentration_risk: float


@dataclass
class ClusterAnalysis:
    """Cluster analysis results"""
    clusters: Dict[str, int]
    n_clusters: int
    cluster_correlations: Dict[int, float]
    inter_cluster_correlation: float
    intra_cluster_correlation: float
    silhouette_score: float


@dataclass
class RollingCorrelation:
    """Rolling correlation analysis"""
    timestamps: List[datetime]
    correlations: List[float]
    volatility: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    breakpoints: List[datetime]


class CorrelationAnalyzer:
    """Advanced Correlation Analysis System"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Analysis parameters
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.rolling_window = self.config.get('rolling_window', 30)  # 30 days
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.min_periods = self.config.get('min_periods', 20)
        
        # Clustering parameters
        self.max_clusters = self.config.get('max_clusters', 10)
        self.linkage_method = self.config.get('linkage_method', 'ward')
        self.distance_threshold = self.config.get('distance_threshold', 0.5)
        
        # Data storage
        self.returns_data: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None
        self.symbols: List[str] = []
        
        # Analysis results
        self.correlation_history: List[CorrelationMetrics] = []
        self.cluster_history: List[ClusterAnalysis] = []
        
        logger.info("CorrelationAnalyzer initialized")
    
    def set_data(self, returns_data: pd.DataFrame, price_data: Optional[pd.DataFrame] = None):
        """Set returns and price data for analysis"""
        try:
            self.returns_data = returns_data.copy()
            self.symbols = list(returns_data.columns)
            
            if price_data is not None:
                self.price_data = price_data.copy()
            
            logger.info(f"Data set for {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error setting data: {e}")
    
    def calculate_correlation_matrix(self, 
                                   method: CorrelationMethod = CorrelationMethod.PEARSON,
                                   window: Optional[int] = None,
                                   min_periods: Optional[int] = None) -> np.ndarray:
        """Calculate correlation matrix using specified method"""
        
        try:
            if self.returns_data is None:
                logger.error("Returns data not available")
                return np.array([])
            
            data = self.returns_data.copy()
            
            # Apply window if specified
            if window:
                data = data.tail(window)
            
            min_periods = min_periods or self.min_periods
            
            if method == CorrelationMethod.PEARSON:
                corr_matrix = data.corr(method='pearson', min_periods=min_periods)
            elif method == CorrelationMethod.SPEARMAN:
                corr_matrix = data.corr(method='spearman', min_periods=min_periods)
            elif method == CorrelationMethod.KENDALL:
                corr_matrix = data.corr(method='kendall', min_periods=min_periods)
            elif method == CorrelationMethod.EXPONENTIAL:
                # Exponentially weighted correlation
                corr_matrix = data.ewm(span=window or 30).corr().iloc[-len(self.symbols):]
            else:
                # Default to Pearson
                corr_matrix = data.corr(method='pearson', min_periods=min_periods)
            
            # Fill NaN values with 0
            corr_matrix = corr_matrix.fillna(0)
            
            return corr_matrix.values
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return np.array([])
    
    def analyze_correlations(self, 
                           method: CorrelationMethod = CorrelationMethod.PEARSON,
                           window: Optional[int] = None) -> CorrelationMetrics:
        """Comprehensive correlation analysis"""
        
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(method, window)
            
            if corr_matrix.size == 0:
                return CorrelationMetrics(
                    correlation_matrix=np.array([]),
                    average_correlation=0.0,
                    max_correlation=0.0,
                    min_correlation=0.0,
                    correlation_stability=0.0,
                    diversification_ratio=0.0,
                    effective_assets=0.0,
                    concentration_risk=0.0
                )
            
            # Extract upper triangle (excluding diagonal)
            n = corr_matrix.shape[0]
            upper_triangle = corr_matrix[np.triu_indices(n, k=1)]
            
            # Basic correlation metrics
            avg_correlation = np.mean(upper_triangle)
            max_correlation = np.max(upper_triangle)
            min_correlation = np.min(upper_triangle)
            
            # Correlation stability (standard deviation of correlations)
            correlation_stability = np.std(upper_triangle)
            
            # Diversification ratio
            # DR = (sum of individual volatilities) / (portfolio volatility)
            if self.returns_data is not None:
                individual_vols = self.returns_data.std().values
                portfolio_vol = np.sqrt(np.dot(individual_vols, np.dot(corr_matrix, individual_vols)) / n)
                avg_individual_vol = np.mean(individual_vols)
                diversification_ratio = avg_individual_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            else:
                diversification_ratio = 1.0
            
            # Effective number of assets (based on correlation structure)
            # Effective assets = 1 + (n-1) * (1 - avg_correlation) / (1 + (n-1) * avg_correlation)
            if avg_correlation < 1.0:
                effective_assets = (1 + (n - 1) * (1 - avg_correlation)) / (1 + (n - 1) * avg_correlation)
            else:
                effective_assets = 1.0
            
            # Concentration risk (based on eigenvalues of correlation matrix)
            eigenvalues = np.linalg.eigvals(corr_matrix)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative eigenvalues
            
            if len(eigenvalues) > 0:
                # Participation ratio
                participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
                concentration_risk = 1.0 - (participation_ratio / n)
            else:
                concentration_risk = 1.0
            
            metrics = CorrelationMetrics(
                correlation_matrix=corr_matrix,
                average_correlation=avg_correlation,
                max_correlation=max_correlation,
                min_correlation=min_correlation,
                correlation_stability=correlation_stability,
                diversification_ratio=diversification_ratio,
                effective_assets=effective_assets,
                concentration_risk=concentration_risk
            )
            
            # Add to history
            self.correlation_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return CorrelationMetrics(
                correlation_matrix=np.array([]),
                average_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0,
                correlation_stability=0.0,
                diversification_ratio=0.0,
                effective_assets=0.0,
                concentration_risk=0.0
            )
    
    def rolling_correlation_analysis(self, 
                                   symbol1: str, 
                                   symbol2: str,
                                   window: int = None,
                                   method: CorrelationMethod = CorrelationMethod.PEARSON) -> RollingCorrelation:
        """Analyze rolling correlation between two symbols"""
        
        try:
            if self.returns_data is None:
                logger.error("Returns data not available")
                return RollingCorrelation([], [], 0.0, 'stable', [])
            
            if symbol1 not in self.symbols or symbol2 not in self.symbols:
                logger.error(f"Symbols {symbol1} or {symbol2} not found in data")
                return RollingCorrelation([], [], 0.0, 'stable', [])
            
            window = window or self.rolling_window
            data = self.returns_data[[symbol1, symbol2]].copy()
            
            # Calculate rolling correlation
            if method == CorrelationMethod.PEARSON:
                rolling_corr = data[symbol1].rolling(window=window).corr(data[symbol2])
            elif method == CorrelationMethod.SPEARMAN:
                rolling_corr = data[symbol1].rolling(window=window).corr(data[symbol2], method='spearman')
            else:
                rolling_corr = data[symbol1].rolling(window=window).corr(data[symbol2])
            
            # Remove NaN values
            rolling_corr = rolling_corr.dropna()
            
            if rolling_corr.empty:
                return RollingCorrelation([], [], 0.0, 'stable', [])
            
            # Extract timestamps and correlations
            timestamps = rolling_corr.index.tolist()
            correlations = rolling_corr.values.tolist()
            
            # Calculate volatility of correlations
            volatility = np.std(correlations)
            
            # Determine trend
            if len(correlations) > 1:
                # Linear regression to determine trend
                x = np.arange(len(correlations))
                slope, _, _, p_value, _ = stats.linregress(x, correlations)
                
                if p_value < 0.05:  # Significant trend
                    if slope > 0.01:
                        trend = 'increasing'
                    elif slope < -0.01:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Detect structural breaks (simplified)
            breakpoints = self._detect_correlation_breakpoints(correlations, timestamps)
            
            return RollingCorrelation(
                timestamps=timestamps,
                correlations=correlations,
                volatility=volatility,
                trend=trend,
                breakpoints=breakpoints
            )
            
        except Exception as e:
            logger.error(f"Error in rolling correlation analysis: {e}")
            return RollingCorrelation([], [], 0.0, 'stable', [])
    
    def cluster_analysis(self, 
                        method: ClusteringMethod = ClusteringMethod.HIERARCHICAL,
                        n_clusters: Optional[int] = None,
                        correlation_method: CorrelationMethod = CorrelationMethod.PEARSON) -> ClusterAnalysis:
        """Perform cluster analysis based on correlations"""
        
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(correlation_method)
            
            if corr_matrix.size == 0:
                return ClusterAnalysis({}, 0, {}, 0.0, 0.0, 0.0)
            
            # Convert correlation to distance
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            if method == ClusteringMethod.HIERARCHICAL:
                return self._hierarchical_clustering(distance_matrix, n_clusters)
            else:
                logger.warning(f"Clustering method {method.value} not implemented")
                return ClusterAnalysis({}, 0, {}, 0.0, 0.0, 0.0)
            
        except Exception as e:
            logger.error(f"Error in cluster analysis: {e}")
            return ClusterAnalysis({}, 0, {}, 0.0, 0.0, 0.0)
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray, 
                                n_clusters: Optional[int] = None) -> ClusterAnalysis:
        """Perform hierarchical clustering"""
        
        try:
            # Convert to condensed distance matrix
            condensed_distances = squareform(distance_matrix)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
            
            # Determine number of clusters
            if n_clusters is None:
                # Use distance threshold
                cluster_labels = fcluster(linkage_matrix, self.distance_threshold, criterion='distance')
                n_clusters = len(np.unique(cluster_labels))
            else:
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Create cluster mapping
            clusters = {symbol: int(label) for symbol, label in zip(self.symbols, cluster_labels)}
            
            # Calculate cluster statistics
            cluster_correlations = {}
            for cluster_id in range(1, n_clusters + 1):
                cluster_symbols = [symbol for symbol, label in clusters.items() if label == cluster_id]
                if len(cluster_symbols) > 1:
                    cluster_indices = [self.symbols.index(symbol) for symbol in cluster_symbols]
                    cluster_corr_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                    # Convert back to correlation
                    cluster_corr_matrix = 1 - 2 * (cluster_corr_matrix ** 2)
                    upper_triangle = cluster_corr_matrix[np.triu_indices(len(cluster_indices), k=1)]
                    cluster_correlations[cluster_id] = np.mean(upper_triangle)
                else:
                    cluster_correlations[cluster_id] = 1.0
            
            # Calculate inter-cluster and intra-cluster correlations
            intra_cluster_corr = np.mean(list(cluster_correlations.values()))
            
            # Inter-cluster correlation
            inter_cluster_pairs = []
            for i in range(len(self.symbols)):
                for j in range(i + 1, len(self.symbols)):
                    if clusters[self.symbols[i]] != clusters[self.symbols[j]]:
                        # Convert distance back to correlation
                        correlation = 1 - 2 * (distance_matrix[i, j] ** 2)
                        inter_cluster_pairs.append(correlation)
            
            inter_cluster_corr = np.mean(inter_cluster_pairs) if inter_cluster_pairs else 0.0
            
            # Calculate silhouette score (simplified)
            silhouette_score = self._calculate_silhouette_score(distance_matrix, cluster_labels)
            
            analysis = ClusterAnalysis(
                clusters=clusters,
                n_clusters=n_clusters,
                cluster_correlations=cluster_correlations,
                inter_cluster_correlation=inter_cluster_corr,
                intra_cluster_correlation=intra_cluster_corr,
                silhouette_score=silhouette_score
            )
            
            # Add to history
            self.cluster_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            return ClusterAnalysis({}, 0, {}, 0.0, 0.0, 0.0)
    
    def _calculate_silhouette_score(self, distance_matrix: np.ndarray, 
                                   cluster_labels: np.ndarray) -> float:
        """Calculate simplified silhouette score"""
        
        try:
            n = len(cluster_labels)
            silhouette_scores = []
            
            for i in range(n):
                # Same cluster distances
                same_cluster = cluster_labels == cluster_labels[i]
                same_cluster[i] = False  # Exclude self
                
                if np.sum(same_cluster) > 0:
                    a_i = np.mean(distance_matrix[i, same_cluster])
                else:
                    a_i = 0
                
                # Different cluster distances
                b_i = float('inf')
                for cluster_id in np.unique(cluster_labels):
                    if cluster_id != cluster_labels[i]:
                        other_cluster = cluster_labels == cluster_id
                        if np.sum(other_cluster) > 0:
                            b_cluster = np.mean(distance_matrix[i, other_cluster])
                            b_i = min(b_i, b_cluster)
                
                if b_i == float('inf'):
                    b_i = 0
                
                # Silhouette score for point i
                if max(a_i, b_i) > 0:
                    s_i = (b_i - a_i) / max(a_i, b_i)
                else:
                    s_i = 0
                
                silhouette_scores.append(s_i)
            
            return np.mean(silhouette_scores)
            
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            return 0.0
    
    def _detect_correlation_breakpoints(self, correlations: List[float], 
                                       timestamps: List[datetime]) -> List[datetime]:
        """Detect structural breaks in correlation time series"""
        
        try:
            if len(correlations) < 10:
                return []
            
            breakpoints = []
            window_size = max(5, len(correlations) // 10)
            
            for i in range(window_size, len(correlations) - window_size):
                # Compare means before and after point i
                before_mean = np.mean(correlations[i-window_size:i])
                after_mean = np.mean(correlations[i:i+window_size])
                
                # Simple threshold-based detection
                if abs(before_mean - after_mean) > 0.3:  # 30% change threshold
                    breakpoints.append(timestamps[i])
            
            return breakpoints
            
        except Exception as e:
            logger.error(f"Error detecting breakpoints: {e}")
            return []
    
    def get_correlation_heatmap_data(self, 
                                   method: CorrelationMethod = CorrelationMethod.PEARSON) -> Dict[str, Any]:
        """Get data for correlation heatmap visualization"""
        
        try:
            corr_matrix = self.calculate_correlation_matrix(method)
            
            if corr_matrix.size == 0:
                return {}
            
            return {
                'correlation_matrix': corr_matrix.tolist(),
                'symbols': self.symbols,
                'method': method.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting heatmap data: {e}")
            return {}
    
    def identify_highly_correlated_pairs(self, 
                                       threshold: float = None,
                                       method: CorrelationMethod = CorrelationMethod.PEARSON) -> List[Tuple[str, str, float]]:
        """Identify pairs of symbols with high correlation"""
        
        try:
            threshold = threshold or self.correlation_threshold
            corr_matrix = self.calculate_correlation_matrix(method)
            
            if corr_matrix.size == 0:
                return []
            
            high_corr_pairs = []
            n = len(self.symbols)
            
            for i in range(n):
                for j in range(i + 1, n):
                    correlation = corr_matrix[i, j]
                    if abs(correlation) >= threshold:
                        high_corr_pairs.append((
                            self.symbols[i],
                            self.symbols[j],
                            correlation
                        ))
            
            # Sort by absolute correlation (descending)
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            return high_corr_pairs
            
        except Exception as e:
            logger.error(f"Error identifying highly correlated pairs: {e}")
            return []
    
    def calculate_portfolio_correlation_risk(self, weights: Dict[str, float],
                                           method: CorrelationMethod = CorrelationMethod.PEARSON) -> Dict[str, float]:
        """Calculate correlation risk metrics for a portfolio"""
        
        try:
            corr_matrix = self.calculate_correlation_matrix(method)
            
            if corr_matrix.size == 0:
                return {}
            
            # Convert weights to array
            weights_array = np.array([weights.get(symbol, 0.0) for symbol in self.symbols])
            
            # Portfolio correlation risk
            portfolio_corr = np.dot(weights_array, np.dot(corr_matrix, weights_array))
            
            # Average pairwise correlation weighted by portfolio weights
            weighted_avg_corr = 0.0
            total_weight_pairs = 0.0
            
            for i in range(len(self.symbols)):
                for j in range(i + 1, len(self.symbols)):
                    weight_product = weights_array[i] * weights_array[j]
                    weighted_avg_corr += weight_product * corr_matrix[i, j]
                    total_weight_pairs += weight_product
            
            if total_weight_pairs > 0:
                weighted_avg_corr /= total_weight_pairs
            
            # Concentration risk based on correlations
            concentration_risk = np.sum(weights_array ** 2)  # Herfindahl index
            
            # Diversification benefit
            individual_risk = np.sum(weights_array ** 2)
            portfolio_risk = portfolio_corr
            diversification_benefit = 1 - (portfolio_risk / individual_risk) if individual_risk > 0 else 0
            
            return {
                'portfolio_correlation': portfolio_corr,
                'weighted_average_correlation': weighted_avg_corr,
                'concentration_risk': concentration_risk,
                'diversification_benefit': diversification_benefit,
                'correlation_risk_score': weighted_avg_corr * concentration_risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation risk: {e}")
            return {}
    
    def generate_correlation_report(self, 
                                  method: CorrelationMethod = CorrelationMethod.PEARSON) -> Dict[str, Any]:
        """Generate comprehensive correlation analysis report"""
        
        try:
            # Basic correlation analysis
            correlation_metrics = self.analyze_correlations(method)
            
            # Cluster analysis
            cluster_analysis = self.cluster_analysis(ClusteringMethod.HIERARCHICAL, correlation_method=method)
            
            # Highly correlated pairs
            high_corr_pairs = self.identify_highly_correlated_pairs(method=method)
            
            # Heatmap data
            heatmap_data = self.get_correlation_heatmap_data(method)
            
            return {
                'correlation_metrics': {
                    'average_correlation': correlation_metrics.average_correlation,
                    'max_correlation': correlation_metrics.max_correlation,
                    'min_correlation': correlation_metrics.min_correlation,
                    'correlation_stability': correlation_metrics.correlation_stability,
                    'diversification_ratio': correlation_metrics.diversification_ratio,
                    'effective_assets': correlation_metrics.effective_assets,
                    'concentration_risk': correlation_metrics.concentration_risk
                },
                'cluster_analysis': {
                    'n_clusters': cluster_analysis.n_clusters,
                    'clusters': cluster_analysis.clusters,
                    'inter_cluster_correlation': cluster_analysis.inter_cluster_correlation,
                    'intra_cluster_correlation': cluster_analysis.intra_cluster_correlation,
                    'silhouette_score': cluster_analysis.silhouette_score
                },
                'high_correlation_pairs': [
                    {'symbol1': pair[0], 'symbol2': pair[1], 'correlation': pair[2]}
                    for pair in high_corr_pairs[:10]  # Top 10 pairs
                ],
                'heatmap_data': heatmap_data,
                'analysis_timestamp': datetime.now().isoformat(),
                'method': method.value,
                'n_symbols': len(self.symbols)
            }
            
        except Exception as e:
            logger.error(f"Error generating correlation report: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get correlation analyzer statistics"""
        return {
            'n_symbols': len(self.symbols),
            'n_correlation_analyses': len(self.correlation_history),
            'n_cluster_analyses': len(self.cluster_history),
            'lookback_period': self.lookback_period,
            'rolling_window': self.rolling_window,
            'correlation_threshold': self.correlation_threshold,
            'max_clusters': self.max_clusters,
            'linkage_method': self.linkage_method,
            'distance_threshold': self.distance_threshold,
            'last_analysis': self.correlation_history[-1] if self.correlation_history else None
        }