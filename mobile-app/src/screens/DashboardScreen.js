import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Dimensions,
} from 'react-native';
import {LineChart} from 'react-native-chart-kit';
import {ApiService} from '../services/ApiService';

const screenWidth = Dimensions.get('window').width;

export default function DashboardScreen() {
  const [dashboardData, setDashboardData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [goldPrice, setGoldPrice] = useState(2000);

  useEffect(() => {
    loadDashboardData();
    
    // Setup real-time price updates
    const interval = setInterval(() => {
      updateGoldPrice();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      const data = await ApiService.getDashboardData();
      setDashboardData(data);
    } catch (error) {
      console.error('Dashboard load error:', error);
    }
  };

  const updateGoldPrice = async () => {
    try {
      const price = await ApiService.getCurrentGoldPrice();
      setGoldPrice(price);
    } catch (error) {
      console.error('Price update error:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const chartData = {
    labels: ['1H', '6H', '12H', '1D', '7D'],
    datasets: [
      {
        data: [1985, 1992, 2001, 2008, goldPrice],
        color: (opacity = 1) => `rgba(255, 215, 0, ${opacity})`,
        strokeWidth: 3,
      },
    ],
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }>
      
      {/* Gold Price Card */}
      <View style={styles.priceCard}>
        <Text style={styles.priceLabel}>XAU/USD</Text>
        <Text style={styles.priceValue}>${goldPrice.toFixed(2)}</Text>
        <Text style={styles.priceChange}>+$12.50 (+0.62%)</Text>
      </View>

      {/* Price Chart */}
      <View style={styles.chartCard}>
        <Text style={styles.cardTitle}>Price Chart</Text>
        <LineChart
          data={chartData}
          width={screenWidth - 40}
          height={220}
          chartConfig={{
            backgroundColor: '#1a1a1a',
            backgroundGradientFrom: '#1a1a1a',
            backgroundGradientTo: '#2a2a2a',
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(255, 215, 0, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
          }}
          style={styles.chart}
        />
      </View>

      {/* Portfolio Summary */}
      <View style={styles.summaryCard}>
        <Text style={styles.cardTitle}>Portfolio Summary</Text>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Balance:</Text>
          <Text style={styles.summaryValue}>$125,450.00</Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Today P&L:</Text>
          <Text style={[styles.summaryValue, styles.profit]}>+$2,340.50</Text>
        </View>
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>Open Positions:</Text>
          <Text style={styles.summaryValue}>3</Text>
        </View>
      </View>

      {/* AI Predictions */}
      <View style={styles.predictionCard}>
        <Text style={styles.cardTitle}>AI Predictions</Text>
        <View style={styles.predictionItem}>
          <Text style={styles.predictionLabel}>Next 1H:</Text>
          <Text style={[styles.predictionValue, styles.bullish]}>BULLISH 87%</Text>
        </View>
        <View style={styles.predictionItem}>
          <Text style={styles.predictionLabel}>Next 4H:</Text>
          <Text style={[styles.predictionValue, styles.neutral]}>NEUTRAL 65%</Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    padding: 20,
  },
  priceCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#FFD700',
  },
  priceLabel: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
  },
  priceValue: {
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
    marginVertical: 8,
  },
  priceChange: {
    color: '#4CAF50',
    fontSize: 16,
  },
  chartCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  cardTitle: {
    color: '#FFD700',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  chart: {
    borderRadius: 8,
  },
  summaryCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  summaryLabel: {
    color: '#ccc',
    fontSize: 16,
  },
  summaryValue: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  profit: {
    color: '#4CAF50',
  },
  predictionCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  predictionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  predictionLabel: {
    color: '#ccc',
    fontSize: 16,
  },
  predictionValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  bullish: {
    color: '#4CAF50',
  },
  neutral: {
    color: '#FF9800',
  },
});