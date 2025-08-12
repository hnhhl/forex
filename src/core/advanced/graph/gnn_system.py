"""
Graph Neural Networks & Advanced AI System
Ultimate XAU Super System V4.0 - Phase 4 Advanced Technologies

Advanced AI with graph neural networks:
- Market relationship modeling with GNNs
- Knowledge graph construction for financial data
- Graph-based predictions
- Explainable AI implementation
- AI interpretability tools
- Network analysis of market dependencies
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import graph and deep learning libraries
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available - using simplified graph operations")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using simplified neural networks")

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in financial knowledge graph"""
    ASSET = "asset"
    ECONOMIC_INDICATOR = "economic_indicator"
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    MARKET_SENTIMENT = "market_sentiment"
    TECHNICAL_INDICATOR = "technical_indicator"


class EdgeType(Enum):
    """Types of edges in financial knowledge graph"""
    CORRELATION = "correlation"
    CAUSATION = "causation"
    INFLUENCE = "influence"
    DEPENDENCY = "dependency"
    SIMILARITY = "similarity"


@dataclass
class GraphNode:
    """Node in financial knowledge graph"""
    node_id: str
    node_type: NodeType
    name: str
    features: np.ndarray
    importance: float
    last_updated: datetime


@dataclass
class GraphEdge:
    """Edge in financial knowledge graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    confidence: float
    temporal: bool = False


@dataclass
class GNNPrediction:
    """Prediction result from GNN"""
    prediction: float
    confidence: float
    contributing_nodes: List[str]
    explanation: str
    attention_weights: Dict[str, float]


class SimpleGNNLayer:
    """Simplified Graph Neural Network layer"""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights (simplified)
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
        
    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Forward pass through GNN layer"""
        # Message passing: aggregate neighbor features
        messages = adjacency_matrix @ node_features
        
        # Apply linear transformation
        output = messages @ self.W + self.b
        
        # Apply activation (ReLU)
        output = np.maximum(0, output)
        
        return output


class FinancialKnowledgeGraph:
    """Financial knowledge graph for market relationships"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.graph = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        
        logger.info("FinancialKnowledgeGraph initialized")
    
    def build_gold_market_graph(self) -> Dict[str, Any]:
        """Build knowledge graph for gold market relationships"""
        try:
            # Add nodes for gold market ecosystem
            nodes_data = [
                ("XAU_USD", NodeType.ASSET, "Gold USD", np.random.randn(10), 1.0),
                ("USD_INDEX", NodeType.ECONOMIC_INDICATOR, "US Dollar Index", np.random.randn(10), 0.9),
                ("FED_RATE", NodeType.ECONOMIC_INDICATOR, "Federal Funds Rate", np.random.randn(10), 0.8),
                ("INFLATION", NodeType.ECONOMIC_INDICATOR, "CPI Inflation", np.random.randn(10), 0.7),
                ("VIX", NodeType.MARKET_SENTIMENT, "Market Volatility", np.random.randn(10), 0.6),
                ("BTC_USD", NodeType.ASSET, "Bitcoin USD", np.random.randn(10), 0.5),
                ("SPY", NodeType.ASSET, "S&P 500 ETF", np.random.randn(10), 0.8),
                ("TNX", NodeType.ASSET, "10-Year Treasury", np.random.randn(10), 0.7),
            ]
            
            # Add nodes
            for node_id, node_type, name, features, importance in nodes_data:
                self.add_node(node_id, node_type, name, features, importance)
            
            # Add relationships (edges)
            relationships = [
                ("USD_INDEX", "XAU_USD", EdgeType.CORRELATION, -0.8, 0.9),  # Strong negative correlation
                ("FED_RATE", "XAU_USD", EdgeType.CAUSATION, -0.7, 0.8),   # Rate hikes hurt gold
                ("INFLATION", "XAU_USD", EdgeType.CORRELATION, 0.6, 0.7),  # Inflation hedge
                ("VIX", "XAU_USD", EdgeType.CORRELATION, 0.4, 0.6),        # Safe haven demand
                ("BTC_USD", "XAU_USD", EdgeType.SIMILARITY, 0.3, 0.5),     # Digital gold narrative
                ("SPY", "XAU_USD", EdgeType.CORRELATION, -0.2, 0.4),       # Risk-on vs safe haven
                ("TNX", "XAU_USD", EdgeType.CAUSATION, -0.6, 0.8),         # Real rates impact
                ("FED_RATE", "TNX", EdgeType.INFLUENCE, 0.8, 0.9),         # Fed controls rates
                ("INFLATION", "FED_RATE", EdgeType.INFLUENCE, 0.7, 0.8),   # Inflation drives policy
            ]
            
            for source, target, edge_type, weight, confidence in relationships:
                self.add_edge(source, target, edge_type, weight, confidence)
            
            # Calculate graph metrics
            metrics = self.calculate_graph_metrics()
            
            return {
                'nodes_count': len(self.nodes),
                'edges_count': len(self.edges),
                'graph_metrics': metrics,
                'centrality_scores': self.calculate_node_centrality(),
                'strongest_relationships': self.get_strongest_relationships(),
                'graph_construction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph construction failed: {e}")
            return {'error': str(e)}
    
    def add_node(self, node_id: str, node_type: NodeType, name: str, 
                features: np.ndarray, importance: float):
        """Add node to knowledge graph"""
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            features=features,
            importance=importance,
            last_updated=datetime.now()
        )
        
        self.nodes[node_id] = node
        
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_node(node_id, **{
                'type': node_type.value,
                'name': name,
                'importance': importance
            })
    
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, 
                weight: float, confidence: float):
        """Add edge to knowledge graph"""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence
        )
        
        self.edges.append(edge)
        
        if NETWORKX_AVAILABLE and self.graph is not None:
            self.graph.add_edge(source_id, target_id, **{
                'type': edge_type.value,
                'weight': weight,
                'confidence': confidence
            })
    
    def calculate_graph_metrics(self) -> Dict[str, float]:
        """Calculate graph topology metrics"""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {'error': 'NetworkX not available'}
        
        try:
            metrics = {
                'density': nx.density(self.graph),
                'average_clustering': nx.average_clustering(self.graph.to_undirected()),
                'number_of_components': nx.number_weakly_connected_components(self.graph),
                'average_path_length': 0.0  # Calculate if connected
            }
            
            # Calculate average path length if graph is connected
            if nx.is_weakly_connected(self.graph):
                metrics['average_path_length'] = nx.average_shortest_path_length(self.graph.to_undirected())
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def calculate_node_centrality(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures for nodes"""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {}
        
        try:
            centrality_measures = {
                'degree_centrality': nx.degree_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph),
                'closeness_centrality': nx.closeness_centrality(self.graph),
                'pagerank': nx.pagerank(self.graph)
            }
            
            # Combine into single score per node
            combined_centrality = {}
            for node_id in self.nodes.keys():
                combined_score = (
                    centrality_measures['degree_centrality'].get(node_id, 0) * 0.3 +
                    centrality_measures['betweenness_centrality'].get(node_id, 0) * 0.3 +
                    centrality_measures['closeness_centrality'].get(node_id, 0) * 0.2 +
                    centrality_measures['pagerank'].get(node_id, 0) * 0.2
                )
                combined_centrality[node_id] = combined_score
            
            centrality_measures['combined_centrality'] = combined_centrality
            return centrality_measures
            
        except Exception as e:
            logger.warning(f"Centrality calculation failed: {e}")
            return {}
    
    def get_strongest_relationships(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get strongest relationships in the graph"""
        relationships = []
        
        for edge in self.edges:
            relationship = {
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.edge_type.value,
                'weight': edge.weight,
                'confidence': edge.confidence,
                'strength': abs(edge.weight) * edge.confidence
            }
            relationships.append(relationship)
        
        # Sort by strength and return top k
        relationships.sort(key=lambda x: x['strength'], reverse=True)
        return relationships[:top_k]


class GraphNeuralNetwork:
    """Graph Neural Network for financial prediction"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build GNN layers
        self.layer1 = SimpleGNNLayer(input_dim, hidden_dim)
        self.layer2 = SimpleGNNLayer(hidden_dim, hidden_dim)
        self.layer3 = SimpleGNNLayer(hidden_dim, output_dim)
        
        # Attention mechanism (simplified)
        self.attention_weights = np.random.randn(hidden_dim) * 0.1
        
        logger.info("GraphNeuralNetwork initialized")
    
    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Forward pass through GNN"""
        # Layer 1
        h1 = self.layer1.forward(node_features, adjacency_matrix)
        
        # Layer 2
        h2 = self.layer2.forward(h1, adjacency_matrix)
        
        # Attention mechanism
        attention_scores = h2 @ self.attention_weights
        attention_weights = self._softmax(attention_scores)
        
        # Weighted aggregation
        attended_features = h2 * attention_weights.reshape(-1, 1)
        
        # Layer 3
        output = self.layer3.forward(attended_features, adjacency_matrix)
        
        # Global pooling for graph-level prediction
        graph_prediction = np.mean(output, axis=0)
        
        # Create attention weights dictionary
        attention_dict = {f"node_{i}": float(w) for i, w in enumerate(attention_weights)}
        
        return graph_prediction, attention_dict
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict_gold_price_movement(self, knowledge_graph: FinancialKnowledgeGraph) -> GNNPrediction:
        """Predict gold price movement using graph neural network"""
        try:
            # Extract node features and adjacency matrix
            node_features = self._extract_node_features(knowledge_graph)
            adjacency_matrix = self._build_adjacency_matrix(knowledge_graph)
            
            # Forward pass
            prediction, attention_weights = self.forward(node_features, adjacency_matrix)
            
            # Convert to interpretable prediction
            price_movement = float(prediction[0])
            confidence = min(1.0, abs(price_movement) * 2)  # Simple confidence measure
            
            # Generate explanation
            explanation = self._generate_explanation(knowledge_graph, attention_weights, price_movement)
            
            # Find contributing nodes
            contributing_nodes = self._find_contributing_nodes(knowledge_graph, attention_weights)
            
            return GNNPrediction(
                prediction=price_movement,
                confidence=confidence,
                contributing_nodes=contributing_nodes,
                explanation=explanation,
                attention_weights=attention_weights
            )
            
        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return GNNPrediction(
                prediction=0.0,
                confidence=0.0,
                contributing_nodes=[],
                explanation=f"Prediction failed: {str(e)}",
                attention_weights={}
            )
    
    def _extract_node_features(self, kg: FinancialKnowledgeGraph) -> np.ndarray:
        """Extract node features as matrix"""
        features = []
        for node in kg.nodes.values():
            features.append(node.features)
        return np.array(features)
    
    def _build_adjacency_matrix(self, kg: FinancialKnowledgeGraph) -> np.ndarray:
        """Build adjacency matrix from knowledge graph"""
        n_nodes = len(kg.nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Create node index mapping
        node_ids = list(kg.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Fill adjacency matrix
        for edge in kg.edges:
            source_idx = node_to_idx.get(edge.source_id)
            target_idx = node_to_idx.get(edge.target_id)
            
            if source_idx is not None and target_idx is not None:
                # Use edge weight as adjacency value
                adj_matrix[source_idx, target_idx] = edge.weight * edge.confidence
        
        # Add self-loops
        np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix
    
    def _generate_explanation(self, kg: FinancialKnowledgeGraph, 
                            attention_weights: Dict[str, float], 
                            prediction: float) -> str:
        """Generate human-readable explanation"""
        direction = "upward" if prediction > 0 else "downward" if prediction < 0 else "sideways"
        strength = "strong" if abs(prediction) > 0.5 else "moderate" if abs(prediction) > 0.2 else "weak"
        
        # Find most important nodes
        node_ids = list(kg.nodes.keys())
        top_nodes = []
        
        for node_key, weight in attention_weights.items():
            if weight > 0.1:  # Only significant contributions
                node_idx = int(node_key.split('_')[1])
                if node_idx < len(node_ids):
                    node_id = node_ids[node_idx]
                    node_name = kg.nodes[node_id].name
                    top_nodes.append((node_name, weight))
        
        top_nodes.sort(key=lambda x: x[1], reverse=True)
        
        explanation = f"GNN predicts {strength} {direction} movement for gold. "
        
        if top_nodes:
            explanation += f"Key factors: {', '.join([node[0] for node in top_nodes[:3]])}."
        
        return explanation
    
    def _find_contributing_nodes(self, kg: FinancialKnowledgeGraph, 
                               attention_weights: Dict[str, float]) -> List[str]:
        """Find nodes that contribute most to prediction"""
        node_ids = list(kg.nodes.keys())
        contributing = []
        
        for node_key, weight in attention_weights.items():
            if weight > 0.1:  # Significant contribution threshold
                node_idx = int(node_key.split('_')[1])
                if node_idx < len(node_ids):
                    contributing.append(node_ids[node_idx])
        
        return contributing


class ExplainableAI:
    """Explainable AI tools for financial predictions"""
    
    def __init__(self):
        logger.info("ExplainableAI initialized")
    
    def explain_prediction(self, prediction: GNNPrediction, 
                          knowledge_graph: FinancialKnowledgeGraph) -> Dict[str, Any]:
        """Generate comprehensive explanation of prediction"""
        try:
            explanation = {
                'prediction_summary': {
                    'direction': 'bullish' if prediction.prediction > 0 else 'bearish' if prediction.prediction < 0 else 'neutral',
                    'magnitude': abs(prediction.prediction),
                    'confidence': prediction.confidence,
                    'strength': 'high' if prediction.confidence > 0.7 else 'medium' if prediction.confidence > 0.4 else 'low'
                },
                'key_factors': self._analyze_key_factors(prediction, knowledge_graph),
                'risk_factors': self._identify_risk_factors(prediction, knowledge_graph),
                'alternative_scenarios': self._generate_alternative_scenarios(prediction),
                'model_confidence': self._assess_model_confidence(prediction),
                'explanation_text': prediction.explanation
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_key_factors(self, prediction: GNNPrediction, 
                           kg: FinancialKnowledgeGraph) -> List[Dict[str, Any]]:
        """Analyze key factors driving the prediction"""
        factors = []
        
        for node_id in prediction.contributing_nodes:
            if node_id in kg.nodes:
                node = kg.nodes[node_id]
                attention_weight = prediction.attention_weights.get(f"node_{list(kg.nodes.keys()).index(node_id)}", 0)
                
                factor = {
                    'factor_name': node.name,
                    'factor_type': node.node_type.value,
                    'importance': node.importance,
                    'attention_weight': attention_weight,
                    'impact': 'positive' if attention_weight > 0 else 'negative'
                }
                factors.append(factor)
        
        # Sort by attention weight
        factors.sort(key=lambda x: x['attention_weight'], reverse=True)
        return factors[:5]  # Top 5 factors
    
    def _identify_risk_factors(self, prediction: GNNPrediction, 
                             kg: FinancialKnowledgeGraph) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if prediction.confidence < 0.5:
            risks.append("Low model confidence - prediction uncertainty is high")
        
        if len(prediction.contributing_nodes) < 3:
            risks.append("Limited factor diversity - prediction based on few inputs")
        
        # Check for conflicting signals
        attention_values = list(prediction.attention_weights.values())
        if len(attention_values) > 1:
            variance = np.var(attention_values)
            if variance > 0.1:
                risks.append("Conflicting signals from different market factors")
        
        return risks
    
    def _generate_alternative_scenarios(self, prediction: GNNPrediction) -> List[Dict[str, Any]]:
        """Generate alternative scenarios"""
        base_prediction = prediction.prediction
        
        scenarios = [
            {
                'scenario': 'base_case',
                'prediction': base_prediction,
                'probability': prediction.confidence
            },
            {
                'scenario': 'optimistic',
                'prediction': base_prediction * 1.5,
                'probability': max(0.1, prediction.confidence - 0.2)
            },
            {
                'scenario': 'pessimistic',
                'prediction': base_prediction * 0.5,
                'probability': max(0.1, prediction.confidence - 0.2)
            }
        ]
        
        return scenarios
    
    def _assess_model_confidence(self, prediction: GNNPrediction) -> Dict[str, Any]:
        """Assess model confidence and reliability"""
        confidence_assessment = {
            'overall_confidence': prediction.confidence,
            'confidence_level': 'high' if prediction.confidence > 0.7 else 'medium' if prediction.confidence > 0.4 else 'low',
            'reliability_factors': [],
            'uncertainty_sources': []
        }
        
        # Reliability factors
        if len(prediction.contributing_nodes) >= 3:
            confidence_assessment['reliability_factors'].append("Multiple contributing factors")
        
        if prediction.confidence > 0.6:
            confidence_assessment['reliability_factors'].append("High model confidence")
        
        # Uncertainty sources
        if prediction.confidence < 0.5:
            confidence_assessment['uncertainty_sources'].append("Low prediction confidence")
        
        if abs(prediction.prediction) < 0.1:
            confidence_assessment['uncertainty_sources'].append("Weak signal strength")
        
        return confidence_assessment


class GraphNeuralNetworkSystem:
    """Main Graph Neural Network system"""
    
    def __init__(self):
        self.knowledge_graph = FinancialKnowledgeGraph()
        self.gnn_model = GraphNeuralNetwork()
        self.explainable_ai = ExplainableAI()
        self.is_active = False
        self.last_update = None
        
        logger.info("GraphNeuralNetworkSystem initialized")
    
    def initialize(self) -> bool:
        """Initialize GNN system"""
        try:
            # Build knowledge graph
            graph_result = self.knowledge_graph.build_gold_market_graph()
            if 'error' in graph_result:
                logger.warning(f"Knowledge graph construction had issues: {graph_result['error']}")
            
            self.is_active = True
            self.last_update = datetime.now()
            logger.info("GraphNeuralNetworkSystem started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphNeuralNetworkSystem: {e}")
            return False
    
    def process(self, data: Any = None) -> Dict[str, Any]:
        """Process GNN analysis and predictions"""
        try:
            if not self.is_active:
                return {'error': 'System not active'}
            
            # Generate GNN prediction
            gnn_prediction = self.gnn_model.predict_gold_price_movement(self.knowledge_graph)
            
            # Generate explanation
            explanation = self.explainable_ai.explain_prediction(gnn_prediction, self.knowledge_graph)
            
            # Graph analysis
            graph_metrics = self.knowledge_graph.calculate_graph_metrics()
            centrality_scores = self.knowledge_graph.calculate_node_centrality()
            
            results = {
                'gnn_prediction': {
                    'prediction': gnn_prediction.prediction,
                    'confidence': gnn_prediction.confidence,
                    'direction': 'bullish' if gnn_prediction.prediction > 0 else 'bearish' if gnn_prediction.prediction < 0 else 'neutral',
                    'contributing_nodes': gnn_prediction.contributing_nodes,
                    'explanation': gnn_prediction.explanation
                },
                'explainable_ai': explanation,
                'graph_analysis': {
                    'graph_metrics': graph_metrics,
                    'centrality_scores': centrality_scores.get('combined_centrality', {}),
                    'strongest_relationships': self.knowledge_graph.get_strongest_relationships()
                },
                'system_info': {
                    'nodes_count': len(self.knowledge_graph.nodes),
                    'edges_count': len(self.knowledge_graph.edges),
                    'model_type': 'Graph_Neural_Network',
                    'interpretability': 'high'
                }
            }
            
            self.last_update = datetime.now()
            return results
            
        except Exception as e:
            logger.error(f"GNN processing failed: {e}")
            return {'error': str(e)}
    
    def cleanup(self) -> bool:
        """Cleanup GNN system"""
        try:
            self.is_active = False
            logger.info("GraphNeuralNetworkSystem stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping GraphNeuralNetworkSystem: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get GNN system status"""
        return {
            'system_name': 'GraphNeuralNetworkSystem',
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'dependencies': {
                'networkx': NETWORKX_AVAILABLE,
                'torch': TORCH_AVAILABLE
            }
        }


def demo_gnn_system():
    """Demo function to test GNN system"""
    print("\n🧠 GRAPH NEURAL NETWORK SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    gnn_system = GraphNeuralNetworkSystem()
    
    if not gnn_system.initialize():
        print("❌ Failed to initialize GNN system")
        return
    
    print("✅ GNN system initialized")
    
    # Process GNN analysis
    results = gnn_system.process()
    
    if 'error' in results:
        print(f"❌ GNN analysis failed: {results['error']}")
        return
    
    # Display results
    print(f"\n📊 GNN PREDICTION RESULTS")
    
    # GNN prediction
    prediction = results['gnn_prediction']
    print(f"\n🧠 GNN PREDICTION:")
    print(f"  Direction: {prediction['direction'].upper()}")
    print(f"  Magnitude: {prediction['prediction']:.3f}")
    print(f"  Confidence: {prediction['confidence']:.3f}")
    print(f"  Contributing factors: {len(prediction['contributing_nodes'])}")
    print(f"  Explanation: {prediction['explanation']}")
    
    # Graph analysis
    graph_analysis = results['graph_analysis']
    print(f"\n📈 GRAPH ANALYSIS:")
    if 'graph_metrics' in graph_analysis and 'density' in graph_analysis['graph_metrics']:
        metrics = graph_analysis['graph_metrics']
        print(f"  Graph density: {metrics['density']:.3f}")
        print(f"  Average clustering: {metrics['average_clustering']:.3f}")
    
    # Top relationships
    if 'strongest_relationships' in graph_analysis:
        print(f"\n🔗 STRONGEST RELATIONSHIPS:")
        for i, rel in enumerate(graph_analysis['strongest_relationships'][:3], 1):
            print(f"  {i}. {rel['source']} → {rel['target']}: {rel['weight']:.2f} ({rel['type']})")
    
    # Explainable AI
    if 'explainable_ai' in results and 'prediction_summary' in results['explainable_ai']:
        explanation = results['explainable_ai']
        print(f"\n🔍 EXPLAINABLE AI:")
        summary = explanation['prediction_summary']
        print(f"  Prediction strength: {summary['strength']}")
        print(f"  Model confidence: {summary['confidence']:.3f}")
        
        if 'key_factors' in explanation and explanation['key_factors']:
            print(f"  Key factors:")
            for factor in explanation['key_factors'][:3]:
                print(f"    • {factor['factor_name']}: {factor['attention_weight']:.3f}")
    
    # System info
    system_info = results['system_info']
    print(f"\n⚙️ SYSTEM INFO:")
    print(f"  Knowledge graph: {system_info['nodes_count']} nodes, {system_info['edges_count']} edges")
    print(f"  Model type: {system_info['model_type']}")
    print(f"  Interpretability: {system_info['interpretability']}")
    
    # Cleanup
    gnn_system.cleanup()
    print("\n✅ GNN demo completed")


if __name__ == "__main__":
    demo_gnn_system() 