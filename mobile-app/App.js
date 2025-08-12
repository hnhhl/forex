import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialIcons';

import DashboardScreen from './src/screens/DashboardScreen';
import TradingScreen from './src/screens/TradingScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({route}) => ({
          tabBarIcon: ({focused, color, size}) => {
            let iconName;
            
            switch (route.name) {
              case 'Dashboard':
                iconName = 'dashboard';
                break;
              case 'Trading':
                iconName = 'trending-up';
                break;
              case 'Portfolio':
                iconName = 'account-balance-wallet';
                break;
              case 'Settings':
                iconName = 'settings';
                break;
            }
            
            return <Icon name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#FFD700',
          tabBarInactiveTintColor: 'gray',
          headerStyle: {
            backgroundColor: '#1a1a1a',
          },
          headerTintColor: '#FFD700',
          tabBarStyle: {
            backgroundColor: '#1a1a1a',
          },
        })}>
        <Tab.Screen name="Dashboard" component={DashboardScreen} />
        <Tab.Screen name="Trading" component={TradingScreen} />
        <Tab.Screen name="Portfolio" component={PortfolioScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}