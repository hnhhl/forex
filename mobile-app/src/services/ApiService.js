class ApiService {
  static BASE_URL = 'http://your-api-url.com/api';
  
  static async getDashboardData() {
    try {
      const response = await fetch(`${this.BASE_URL}/dashboard`);
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      // Return mock data for development
      return {
        balance: 125450.00,
        todayPnl: 2340.50,
        openPositions: 3
      };
    }
  }
  
  static async getCurrentGoldPrice() {
    try {
      const response = await fetch(`${this.BASE_URL}/price/xauusd`);
      const data = await response.json();
      return data.price;
    } catch (error) {
      // Return mock price
      return 2000 + Math.random() * 20;
    }
  }
  
  static async getPortfolio() {
    try {
      const response = await fetch(`${this.BASE_URL}/portfolio`);
      return await response.json();
    } catch (error) {
      return {
        positions: [],
        totalValue: 125450.00,
        todayChange: 2340.50
      };
    }
  }
  
  static async placeTrade(tradeData) {
    try {
      const response = await fetch(`${this.BASE_URL}/trades`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(tradeData),
      });
      return await response.json();
    } catch (error) {
      throw error;
    }
  }
}

export {ApiService};