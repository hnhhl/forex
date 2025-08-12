#!/usr/bin/env python3
"""
Kiểm tra chi tiết MT5 connection và data availability
"""

import MetaTrader5 as mt5
import pandas as pd

def check_mt5_detailed():
    """Kiểm tra chi tiết MT5 connection"""
    print('🔍 KIỂM TRA MT5 CONNECTION CHI TIẾT')
    print('=' * 50)

    # Initialize
    if mt5.initialize():
        print('✅ MT5 initialized successfully')
        
        # Check account info
        account_info = mt5.account_info()
        if account_info:
            print(f'📊 Account: {account_info.login}')
            print(f'💰 Balance: {account_info.balance}')
            print(f'🏢 Server: {account_info.server}')
        else:
            print('⚠️ No account info - demo mode')
        
        # Check available symbols
        symbols = mt5.symbols_get()
        if symbols:
            print(f'📈 Available symbols: {len(symbols)}')
            xau_symbols = [s.name for s in symbols if 'XAU' in s.name]
            print(f'🥇 XAU symbols: {xau_symbols[:5]}')
        
        # Try different symbols
        test_symbols = ['XAUUSD', 'XAUUSDc', 'GOLD', 'XAUUSDC']
        working_symbol = None
        
        for symbol in test_symbols:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
            if rates is not None:
                print(f'✅ {symbol}: {len(rates)} rates available')
                df = pd.DataFrame(rates)
                print(f'   Columns: {list(df.columns)}')
                working_symbol = symbol
                
                # Show sample data
                if len(df) > 0:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    print(f'   Sample data:')
                    print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].head(2))
                break
            else:
                print(f'❌ {symbol}: No data')
        
        mt5.shutdown()
        return working_symbol
        
    else:
        print('❌ MT5 initialization failed')
        print(f'Error: {mt5.last_error()}')
        return None

if __name__ == "__main__":
    working_symbol = check_mt5_detailed()
    if working_symbol:
        print(f"\n✅ Có thể sử dụng symbol: {working_symbol}")
    else:
        print(f"\n❌ Không có symbol nào hoạt động") 