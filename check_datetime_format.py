import pandas as pd

# Check H1 data
print("=== H1 DATA ===")
data_h1 = pd.read_csv('data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv')
print("Sample times:")
print(data_h1['time'].head(10).tolist())
print(data_h1['time'].tail(10).tolist())

# Check H4 data  
print("\n=== H4 DATA ===")
data_h4 = pd.read_csv('data/maximum_mt5_v2/XAUUSDc_H4_20250618_115847.csv')
print("Sample times:")
print(data_h4['time'].head(5).tolist())
print(data_h4['time'].tail(5).tolist())

# Check D1 data
print("\n=== D1 DATA ===")
data_d1 = pd.read_csv('data/maximum_mt5_v2/XAUUSDc_D1_20250618_115847.csv')
print("Sample times:")
print(data_d1['time'].head(5).tolist())
print(data_d1['time'].tail(5).tolist()) 