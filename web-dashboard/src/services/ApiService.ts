/**
 * API Service for Ultimate XAU Super System V4.0
 * Handles all communication with backend systems
 */

export interface SystemMetrics {
  performance_boost: number;
  test_coverage: number;
  response_time: number;
  uptime: number;
  active_systems: number;
  total_trades: number;
  portfolio_value: number;
  daily_pnl: number;
  timestamp: string;
}

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  timestamp: string;
  ai_analysis: {
    neural_ensemble: number;
    reinforcement_learning: number;
    meta_learning: number;
    sentiment_analysis: number;
  };
}

export interface SystemHealth {
  overall_status: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'CRITICAL';
  subsystems: {
    trading: string;
    ai_ml: string;
    risk_management: string;
    analysis: string;
    advanced_tech: string;
  };
  last_check: string;
}

class ApiService {
  private baseUrl: string;
  private wsUrl: string;
  private apiKey: string;
  private ws: WebSocket | null = null;

  constructor() {
    // These would normally come from environment variables
    this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
    this.wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
    this.apiKey = process.env.REACT_APP_API_KEY || 'dev-key';
  }

  // Authentication
  private getHeaders(): HeadersInit {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      'X-System-Version': 'v4.0',
    };
  }

  // Generic API call method
  private async apiCall<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.getHeaders(),
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`API call failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API call to ${endpoint} failed:`, error);
      throw error;
    }
  }

  // System Metrics
  async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      const metrics = await this.apiCall<SystemMetrics>('/system/metrics');
      return metrics;
    } catch (error) {
      // Fallback to mock data for development
      console.warn('Using mock data for system metrics');
      return {
        performance_boost: 125.9,
        test_coverage: 90.1,
        response_time: 48.7,
        uptime: 99.90,
        active_systems: 22,
        total_trades: 1247,
        portfolio_value: 1250000,
        daily_pnl: 15420,
        timestamp: new Date().toISOString()
      };
    }
  }

  // Trading Signals
  async getLatestSignals(symbol: string = 'XAUUSD'): Promise<TradingSignal[]> {
    try {
      const signals = await this.apiCall<TradingSignal[]>(`/trading/signals?symbol=${symbol}`);
      return signals;
    } catch (error) {
      console.warn('Using mock data for trading signals');
      return [
        {
          symbol: 'XAUUSD',
          action: 'BUY',
          confidence: 0.85,
          entry_price: 1985.20,
          stop_loss: 1975.00,
          take_profit: 2000.00,
          timestamp: new Date().toISOString(),
          ai_analysis: {
            neural_ensemble: 0.88,
            reinforcement_learning: 0.82,
            meta_learning: 0.85,
            sentiment_analysis: 0.75
          }
        }
      ];
    }
  }

  // Portfolio Data
  async getPortfolioData(): Promise<any> {
    try {
      const portfolio = await this.apiCall<any>('/portfolio/summary');
      return portfolio;
    } catch (error) {
      console.warn('Using mock data for portfolio');
      return {
        total_value: 1250000,
        daily_pnl: 15420,
        positions: [
          {
            symbol: 'XAUUSD',
            size: 2.5,
            entry_price: 1980.50,
            current_price: 1985.20,
            pnl: 1175.00,
            pnl_percentage: 0.24
          },
          {
            symbol: 'BTCUSD',
            size: 0.1,
            entry_price: 42500,
            current_price: 43100,
            pnl: 600.00,
            pnl_percentage: 1.41
          }
        ]
      };
    }
  }

  // System Health
  async getSystemHealth(): Promise<SystemHealth> {
    try {
      const health = await this.apiCall<SystemHealth>('/system/health');
      return health;
    } catch (error) {
      console.warn('Using mock data for system health');
      return {
        overall_status: 'EXCELLENT',
        subsystems: {
          trading: 'OPERATIONAL',
          ai_ml: 'EXCELLENT',
          risk_management: 'OPERATIONAL',
          analysis: 'EXCELLENT',
          advanced_tech: 'OPERATIONAL'
        },
        last_check: new Date().toISOString()
      };
    }
  }

  // Real-time WebSocket connection
  connectWebSocket(onMessage: (data: any) => void, onError?: (error: Event) => void): void {
    try {
      this.ws = new WebSocket(this.wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected to Ultimate XAU System');
        // Send authentication
        this.ws?.send(JSON.stringify({
          type: 'auth',
          token: this.apiKey
        }));
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          this.connectWebSocket(onMessage, onError);
        }, 5000);
      };

    } catch (error) {
      console.error('Failed to establish WebSocket connection:', error);
      onError?.(error as Event);
    }
  }

  // Disconnect WebSocket
  disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // Trading Actions
  async executeOrder(order: {
    symbol: string;
    action: 'BUY' | 'SELL';
    volume: number;
    order_type: 'MARKET' | 'LIMIT' | 'STOP';
    price?: number;
    stop_loss?: number;
    take_profit?: number;
  }): Promise<any> {
    try {
      const result = await this.apiCall('/trading/execute', {
        method: 'POST',
        body: JSON.stringify(order)
      });
      return result;
    } catch (error) {
      console.error('Order execution failed:', error);
      throw error;
    }
  }

  // Risk Management
  async getRiskMetrics(): Promise<any> {
    try {
      const metrics = await this.apiCall('/risk/metrics');
      return metrics;
    } catch (error) {
      console.warn('Using mock data for risk metrics');
      return {
        var_1d: 0.025,
        var_5d: 0.055,
        max_drawdown: 0.018,
        sharpe_ratio: 4.2,
        portfolio_risk: 0.032,
        position_limits: {
          used: 0.65,
          available: 0.35
        }
      };
    }
  }

  // AI Model Performance
  async getAIPerformance(): Promise<any> {
    try {
      const performance = await this.apiCall('/ai/performance');
      return performance;
    } catch (error) {
      console.warn('Using mock data for AI performance');
      return {
        neural_ensemble: {
          accuracy: 0.892,
          precision: 0.885,
          recall: 0.901,
          f1_score: 0.893
        },
        reinforcement_learning: {
          cumulative_reward: 15420,
          win_rate: 0.847,
          avg_trade_duration: 2.5
        },
        meta_learning: {
          adaptation_speed: 0.756,
          knowledge_transfer: 0.823,
          few_shot_accuracy: 0.798
        }
      };
    }
  }

  // System Configuration
  async getSystemConfig(): Promise<any> {
    try {
      const config = await this.apiCall('/system/config');
      return config;
    } catch (error) {
      console.warn('Using default system configuration');
      return {
        trading: {
          max_positions: 5,
          risk_per_trade: 0.02,
          auto_trading: true
        },
        ai: {
          model_update_frequency: 24,
          ensemble_size: 10,
          learning_rate: 0.001
        },
        risk: {
          max_drawdown: 0.05,
          var_confidence: 0.95,
          stress_test_enabled: true
        }
      };
    }
  }

  // Phase 1 Implementation Status
  async getPhase1Status(): Promise<any> {
    try {
      const status = await this.apiCall('/implementation/phase1');
      return status;
    } catch (error) {
      // Mock Phase 1 status for development
      return {
        web_dashboard: { status: 'ACTIVE', completion: 100 },
        monitoring: { status: 'IN_PROGRESS', completion: 60 },
        security: { status: 'PENDING', completion: 0 },
        cloud_deploy: { status: 'PENDING', completion: 0 },
        quantum_hw: { status: 'PENDING', completion: 0 },
        overall_completion: 32
      };
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default ApiService; 