/**
 * Real-time Data Hook for Ultimate XAU Super System V4.0
 * Handles WebSocket connections and real-time data updates
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from '../services/ApiService';

interface UseRealTimeDataOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface RealTimeDataState<T = any> {
  data: T | null;
  isConnected: boolean;
  error: string | null;
  lastUpdate: Date | null;
  connectionAttempts: number;
}

export function useRealTimeData<T = any>(
  endpoint: string, 
  options: UseRealTimeDataOptions = {}
) {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10
  } = options;

  const [state, setState] = useState<RealTimeDataState<T>>({
    data: null,
    isConnected: false,
    error: null,
    lastUpdate: null,
    connectionAttempts: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const messageData = JSON.parse(event.data);
      
      // Filter messages for our specific endpoint
      if (messageData.endpoint === endpoint || endpoint === '/api/realtime/metrics') {
        if (mountedRef.current) {
          setState(prev => ({
            ...prev,
            data: messageData.data || messageData,
            lastUpdate: new Date(),
            error: null
          }));
        }
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      if (mountedRef.current) {
        setState(prev => ({
          ...prev,
          error: 'Failed to parse incoming data'
        }));
      }
    }
  }, [endpoint]);

  // Handle WebSocket connection open
  const handleOpen = useCallback(() => {
    console.log(`WebSocket connected for endpoint: ${endpoint}`);
    if (mountedRef.current) {
      setState(prev => ({
        ...prev,
        isConnected: true,
        error: null,
        connectionAttempts: 0
      }));
    }

    // Subscribe to specific endpoint
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        endpoint: endpoint
      }));
    }
  }, [endpoint]);

  // Handle WebSocket errors
  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
    if (mountedRef.current) {
      setState(prev => ({
        ...prev,
        error: 'Connection error occurred',
        isConnected: false
      }));
    }
  }, []);

  // Handle WebSocket connection close
  const handleClose = useCallback(() => {
    console.log(`WebSocket disconnected for endpoint: ${endpoint}`);
    if (mountedRef.current) {
      setState(prev => ({
        ...prev,
        isConnected: false
      }));
    }

    // Attempt to reconnect if enabled and within limits
    if (autoReconnect && 
        state.connectionAttempts < maxReconnectAttempts && 
        mountedRef.current) {
      
      setState(prev => ({
        ...prev,
        connectionAttempts: prev.connectionAttempts + 1
      }));

      reconnectTimeoutRef.current = setTimeout(() => {
        if (mountedRef.current) {
          connect();
        }
      }, reconnectInterval);
    }
  }, [endpoint, autoReconnect, maxReconnectAttempts, reconnectInterval, state.connectionAttempts]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    try {
      // Clean up existing connection
      if (wsRef.current) {
        wsRef.current.removeEventListener('open', handleOpen);
        wsRef.current.removeEventListener('message', handleMessage);
        wsRef.current.removeEventListener('error', handleError);
        wsRef.current.removeEventListener('close', handleClose);
        wsRef.current.close();
      }

      // Create new WebSocket connection
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
      wsRef.current = new WebSocket(wsUrl);

      // Add event listeners
      wsRef.current.addEventListener('open', handleOpen);
      wsRef.current.addEventListener('message', handleMessage);
      wsRef.current.addEventListener('error', handleError);
      wsRef.current.addEventListener('close', handleClose);

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      if (mountedRef.current) {
        setState(prev => ({
          ...prev,
          error: 'Failed to create connection',
          isConnected: false
        }));
      }
    }
  }, [handleOpen, handleMessage, handleError, handleClose]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.removeEventListener('open', handleOpen);
      wsRef.current.removeEventListener('message', handleMessage);
      wsRef.current.removeEventListener('error', handleError);
      wsRef.current.removeEventListener('close', handleClose);
      wsRef.current.close();
      wsRef.current = null;
    }

    if (mountedRef.current) {
      setState(prev => ({
        ...prev,
        isConnected: false,
        connectionAttempts: 0
      }));
    }
  }, [handleOpen, handleMessage, handleError, handleClose]);

  // Send message through WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // Effect to establish connection on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    // Cleanup on unmount
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Fallback to polling for development when WebSocket is not available
  useEffect(() => {
    let pollingInterval: NodeJS.Timeout | null = null;

    if (!state.isConnected && endpoint === '/api/realtime/metrics') {
      // Start polling every 5 seconds as fallback
      pollingInterval = setInterval(async () => {
        try {
          const metrics = await apiService.getSystemMetrics();
          if (mountedRef.current) {
            setState(prev => ({
              ...prev,
              data: { metrics } as T,
              lastUpdate: new Date(),
              error: null
            }));
          }
        } catch (error) {
          console.error('Polling failed:', error);
        }
      }, 5000);

      // Initial data fetch
      apiService.getSystemMetrics().then(metrics => {
        if (mountedRef.current) {
          setState(prev => ({
            ...prev,
            data: { metrics } as T,
            lastUpdate: new Date()
          }));
        }
      }).catch(error => {
        console.error('Initial data fetch failed:', error);
      });
    }

    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [state.isConnected, endpoint]);

  return {
    data: state.data,
    isConnected: state.isConnected,
    error: state.error,
    lastUpdate: state.lastUpdate,
    connectionAttempts: state.connectionAttempts,
    connect,
    disconnect,
    sendMessage
  };
}

// Specialized hook for system metrics
export function useSystemMetrics() {
  return useRealTimeData('/api/realtime/metrics', {
    autoReconnect: true,
    reconnectInterval: 3000,
    maxReconnectAttempts: 15
  });
}

// Specialized hook for trading signals
export function useTradingSignals(symbol: string = 'XAUUSD') {
  return useRealTimeData(`/api/realtime/signals/${symbol}`, {
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 10
  });
}

// Specialized hook for portfolio updates
export function usePortfolioUpdates() {
  return useRealTimeData('/api/realtime/portfolio', {
    autoReconnect: true,
    reconnectInterval: 2000,
    maxReconnectAttempts: 20
  });
} 