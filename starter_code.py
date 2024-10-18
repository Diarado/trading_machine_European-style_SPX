import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
import bisect 
from collections import deque
from random import randint
from scipy.stats import percentileofscore

class Strategy:
    def __init__(self, start_date, end_date, options_data, underlying) -> None:
        self.capital : float = 100_000_000
        self.portfolio_value : float = 0
        self.start_date : datetime = start_date
        self.end_date : datetime = end_date
        self.options : pd.DataFrame = pd.read_csv(options_data).copy()
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])
        self.spx_minute_data: pd.DataFrame = pd.read_csv(underlying).copy()
        
        self.spx_minute_data['price'] = self.spx_minute_data['price'] / 100
        # Convert the ms_of_day into a time format (milliseconds to timedelta)
        self.spx_minute_data['ms_of_day'] = pd.to_timedelta(self.spx_minute_data['ms_of_day'], unit='ms')
        # Convert ms_of_day from ET to UTC by adding 5 hours
        self.spx_minute_data['ms_of_day'] += pd.Timedelta(hours=5)
        # Convert the 'date' column to datetime format ('YYYYMMDD' format) in UTC
        self.spx_minute_data['date'] = pd.to_datetime(self.spx_minute_data['date'], format='%Y%m%d', utc=True)
        self.spx_minute_data['date'] = self.spx_minute_data['date'].dt.tz_localize(None)
        
        self.spx_minute_data['log_return'] = np.log(self.spx_minute_data['price'] / self.spx_minute_data['price'].shift(1))
        window_size = 26
        self.spx_minute_data['volatility'] = self.spx_minute_data['log_return'].rolling(window=window_size).std() * np.sqrt(window_size)
        self.spx_minute_data = self.spx_minute_data.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        self.spx_minute_data = self.spx_minute_data.dropna()
        self.spx_minute_data['volatility'] = self.spx_minute_data['volatility']*1000
        self.idx = 0
        self.volatility_series = self.spx_minute_data['volatility']
    
    def detect_market_signal(self, macd_value, macd_signal, vol, volatility_series):
        vol_percentile = percentileofscore(volatility_series, vol)
        if macd_value > macd_signal:
            return 'bull'
        elif macd_value < macd_signal:
            return 'bear'
        elif vol_percentile >= 96.77:
            return 'high_volatility'
        return 'neutral'

    def calculate_order_size(self, signals, max_order_size, portfolio_value):
        combined_signal = signals.sum(axis=1)
        order_size = (np.abs(combined_signal) / np.max(np.abs(combined_signal))) * (portfolio_value * 0.01)
        return np.minimum(order_size, max_order_size)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        excess_returns = returns - risk_free_rate / 252  # daily returns
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_max_drawdown(self, equity_curve):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def black_scholes_call(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calculate_implied_volatility(self, S, K, T, r, market_price, option_type):
        def objective(sigma):
            if option_type == 'Call':
                return self.black_scholes_call(S, K, T, r, sigma) - market_price
            else:
                return self.black_scholes_put(S, K, T, r, sigma) - market_price

        sigma_lower = 1e-6
        sigma_upper = 10  # Volatility is between 0.0001% and 1000%

        try:
            implied_vol = brentq(objective, sigma_lower, sigma_upper)
        except ValueError:
            implied_vol = 1e-6
        return implied_vol

    def compute_option_greeks(self, S, K, T, r, sigma, option_type):
        try:
            delta = self.calculate_delta(S, K, T, r, sigma, option_type)
            theta = self.calculate_theta(S, K, T, r, sigma, option_type)
            gamma = self.calculate_gamma(S, K, T, r, sigma)
            vega = self.calculate_vega(S, K, T, r, sigma)
            rho = self.calculate_rho(S, K, T, r, sigma, option_type)

            greeks = {
                'delta': delta,
                'theta': theta,
                'gamma': gamma,
                'vega': vega,
                'rho': rho
            }
            return greeks
        except:
            return None

    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    def calculate_order_size(self, option_premium, bid_size, ask_size, action) -> int:
        # Define risk parameters
        risk_per_trade_percentage = 0.01  # 1% of capital
        max_risk_amount = self.capital * risk_per_trade_percentage
        # Calculate the maximum order size based on risk
        max_order_size = max_risk_amount / (option_premium * 100 + 0.1 * option_premium * 100)

        # Adjust for available bid/ask sizes
        if action == 'buy':
            available_size = ask_size
        else:
            available_size = bid_size

        order_size = min(max_order_size, available_size)
        order_size = int(order_size)
        order_size = max(order_size, 1)
        return order_size
        
    def create_order(self, timestamp, option_symbol, action, option_premium, bid_size, ask_size):
        # Calculate order size
        order_size = self.calculate_order_size(option_premium, bid_size, ask_size, action)
        # print('!!!')
        # print(type(timestamp))
        # print(timestamp)
        
        order = {
            'datetime': timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')+ f'{timestamp.nanosecond:03d}' + 'Z',
            'option_symbol': option_symbol,
            'action': 'B' if action == 'buy' else 'S',
            'order_size': order_size
        }
        # print(order)
        return order
    
    # return a list of parsed options with the closest strike_price
    # example: [  ...strike_price = P...]
    #          [  ..strike_proce = P...] 

    def find_closest_strike(self, cur_expire_date, current_price, options_today):
        available_strikes = options_today[options_today['expiration_date'] == cur_expire_date].copy()  # Create a copy
        if available_strikes.empty:
            return pd.Series()
        else:
            available_strikes['price_diff'] = (available_strikes['strike_price'] - current_price).abs()
            # Find the index of the row with the minimum difference
            min_index = available_strikes['price_diff'].idxmin()
            
            # Return the row with the minimum strike price difference
            return available_strikes.loc[min_index]
        
    def parse_symbol(self, symbol: str):
        pattern = r'(\d{6})([CP])(\d{8})'
        match = re.search(pattern, symbol)
        if match:
            exp, type, strike = match.groups()
            exp_date = datetime.strptime(exp, '%y%m%d').date()
            option_type = 'Call' if type == 'C' else 'Put'
            strike_price = int(strike) / 100000.0
            return exp_date, option_type, strike_price
        else:
            return None, None, None  
        
    def parse_symbol_new(self, symbol: str):
        """
        EXAMPLE: SPX 20230120P2800000
        """
       
        try:
            numbers = symbol.split(" ")[1]
            date = numbers[:8]
            date_yymmdd = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
            action = numbers[8]
            strike_price = int(numbers[9:]) / 1000
            expiration_date = datetime.strptime(date_yymmdd, "%Y-%m-%d").date()
            option_type = 'Call' if action == 'C' else 'Put'
            return expiration_date, option_type, strike_price
        except (IndexError, ValueError):
            return None, None, None
        
    def generate_orders(self) -> pd.DataFrame:
        #parsed_options = self.parse_data(self.options)
        parsed_options = self.options
        risk_free_rate = 0.03  # predetermined risk-free rate
        max_order_size = 100   # Adjustable maximum order size
        stop_loss_pct = 0.05   # Stop loss percentage
        take_profit_pct = 0.10 # Take profit percentage
        order_len = 0
        
        orders = []
        portfolio_value = self.capital
        daily_returns = []
        
        # Window for analyzing recent 26 price movements
        win_len = 26
        window = deque(self.spx_minute_data['price'].iloc[:27])
        # window_options = [] # (timestamp, [timestamp, ..... strike_price, ...])
        
        #     'timestamp',
        #     'instrument_id',
        #     'option_symbol',
        #     'expiration_date',
        #     'option_type',
        #     'strike_price',
        #     'bid_price',
        #     'ask_price',
        #     'bid_size',
        #     'ask_size',
        #     'mid_price',
        #     'spread'
        # initialize:
        # ii = 0
        # while ii < len(parsed_options):
        #     line = parsed_options[ii]
        #     if line['timestamp'] >= pd.Timestamp(2024, 1, 2) + pd.Timedelta(milliseconds=35760000):
        #         break  
        #     window_options.append((line['timestamp'], line))
        #     ii += 1

        cur_idx = 0  # Index pointer for parsed_options
        num_options = len(parsed_options)  # Total number of options
        
        for i, line in self.spx_minute_data.iloc[26:].iterrows():
            minute = line['ms_of_day']
            date = line['date']
            current_price = line['price']
            
            current_time = pd.to_datetime(date) + pd.to_timedelta(minute, unit='ms')

            try:
                macd, signal = self.calculate_macd(window)
            except ValueError:
                print("MACD calculation error")
                continue
            
            options_today = [] # it's actually options to_minute
            window.append(current_price)
            
            while len(window) > win_len:
                window.popleft()
            
            # Compute the threshold time for 26-minute window in milliseconds
            # thres = current_time - pd.Timedelta(minutes=26)
            
            # Use bisect on the timestamp (first element of the tuple in window_options)
            # idx = bisect.bisect_right([x[0] for x in window_options], thres)
            
            # Remove options that are older than the threshold time
            # window_options = window_options[idx:]
            
            parsed_options.rename(columns={
                'ts_recv': 'timestamp',
                'bid_px_00': 'bid_price',
                'ask_px_00': 'ask_price',
                'bid_sz_00': 'bid_size',
                'ask_sz_00': 'ask_size',
                'symbol': 'option_symbol'
            }, inplace=True)
            parsed_options['timestamp'] = pd.to_datetime(parsed_options['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
            
            parsed_columns = parsed_options['option_symbol'].apply(self.parse_symbol).apply(pd.Series)

            # Assign the new columns to the original DataFrame
            parsed_options[['expiration_date', 'option_type', 'strike_price']] = parsed_columns
            
            while cur_idx < num_options and parsed_options['timestamp'].iloc[cur_idx] < current_time:
                cur_idx += 1
                
            while cur_idx < num_options and parsed_options['timestamp'].iloc[cur_idx] < current_time + pd.Timedelta(milliseconds=6000): # all options within 1 min
                options_today.append(parsed_options.iloc[cur_idx]) 
                cur_idx += 1
            
            options_today = pd.DataFrame(options_today)     
            
            
            # use model to do determine which option to buy from options_today (tominute)
            for j, option in options_today.iterrows(): 
                # print(type(option))
                # Assuming 'option' is a pd.Series
                expiration_date, option_type, strike_price = self.parse_symbol(option['option_symbol'])
                # print(expiration_date, option_type, strike_price)
                option['expiration_date'] = expiration_date
                option['option_type'] = option_type
                option['strike_price'] = float(strike_price)

                option['mid_price'] = (float(option['bid_price']) + float(option['ask_price'])) / 2
                option['bid_price'] = float(option['bid_price'])
                option['ask_price'] = float(option['ask_price'])
                option['bid_size'] = int(option['bid_size'])
                option['ask_size'] = int(option['ask_size'])
                print('option:')
                print(option)
                if order_len >= 1000:
                    df_orders = pd.DataFrame(orders)
                    df_orders.to_csv('orders.csv', index=False)
                    return df_orders
                expiration_date = pd.to_datetime(option['expiration_date'])
                timestamp = pd.to_datetime(option['timestamp'])
                days_to_expiration = (expiration_date - timestamp).days

                if days_to_expiration <= 0:
                    continue

                T = days_to_expiration / 365.0

                implied_vol = self.calculate_implied_volatility(current_price, option['strike_price'],
                                                                T, risk_free_rate, option['mid_price'],
                                                                option['option_type'])
                greeks = self.compute_option_greeks(current_price, option['strike_price'],
                                                    T, risk_free_rate, implied_vol,
                                                    option['option_type'])

                if greeks:
                    greeks['option_id'] = option['instrument_id']
                    greeks['option_symbol'] = option['option_symbol']

                # Extract the latest MACD and signal values
                macd_value = macd.iloc[-1]
                macd_signal = signal.iloc[-1]
                # print(f"MACD Value: {macd_value}, Signal Value: {macd_signal}")

                market_signal = self.detect_market_signal(macd_value, macd_signal, days_to_expiration, self.volatility_series)

                timestamp = option['timestamp']
                option_symbol = option['option_symbol']
                cur_expire_date = option['expiration_date']
                delta = greeks['delta']
                option_premium = option['mid_price']
                bid_size = option['bid_size']
                ask_size = option['ask_size']
                if market_signal == 'bull':
                    # Bull Call Spread (buy call with lower strike, sell call with higher strike)
                    
                    if greeks['delta'] > 0.3:
                        # Buy call with lower strike
   
                        order_buy = self.create_order(timestamp, option_symbol, 'buy', option_premium, bid_size, ask_size)
                        orders.append(order_buy)
                        order_len += 1
                        
                        # Sell call with higher strike
                        
                        row = self.find_closest_strike(cur_expire_date, current_price + 2, options_today)  # Arbitrary +2 strike difference
                        if not row.empty:
                            # delta_hedge = self.calculate_delta(current_price, row['strike_price'],
                            #                                         T, risk_free_rate, row['mid_price'],
                            #                                         row['option_type'])
                            order_sell = self.create_order(row['timestamp'], row['option_symbol'], 'sell', option_premium, bid_size, ask_size)
                            orders.append(order_sell)
                            order_len += 1

                elif market_signal == 'bear':
                    # Bear Put Spread (buy put with higher strike, sell put with lower strike)
                    if greeks['delta'] < -0.3:
                        # Buy put with higher strike
                    
                        order_buy = self.create_order(timestamp, option_symbol, 'buy', option_premium, bid_size, ask_size)
                        orders.append(order_buy)
                        order_len += 1
                        
                        # Sell put with lower strike
                        print('options_today')
                        print(options_today)
                        row = self.find_closest_strike(cur_expire_date, current_price - 2, options_today)  # Arbitrary -2 strike difference
                        if not row.empty:
                            # delta_hedge = self.calculate_delta(current_price, row['strike_price'],
                            #                                         T, risk_free_rate, row['mid_price'],
                            #                                         row['option_type'])
                            order_sell = self.create_order(row['timestamp'], row['option_symbol'], 'sell', option_premium, bid_size, ask_size)
                            orders.append(order_sell)
                            order_len += 1

                # elif market_signal == 'near_expiration':
                #     # Iron Condor (sell OTM call and put, buy deeper OTM call and put)
                    
                #     if greeks['delta'] > 0.3:
                #     # Sell OTM call and buy deeper OTM call

                #         row = self.find_closest_strike(current_price + 1, options_today)
                      
                #         option_symbol = line['option_symbol']
                #         order_sell = self.create_order(option_symbol, 'sell', greeks, otm_strike, 'call')
                #         orders.append(order_sell)
                            
                #         deeper_otm_strike_lst = self.find_closest_strike(current_price + 2, options_today)
                #         for line in deeper_otm_strike_lst:
                #             option_symbol = line['option_symbol']
                #             order_buy = self.create_order(option_symbol, 'buy', greeks, deeper_otm_strike, 'call')
                #             orders.append(order_buy)
                            
                #     elif greeks['delta'] < -0.3:
                #         # Sell OTM put and buy deeper OTM put
                #         otm_strike = self.find_closest_strike(current_price - 1, options_today['strike_price'].unique())
                #         deeper_otm_strike = self.find_closest_strike(current_price - 2, options_today['strike_price'].unique())
                #         order_sell = self.create_order(option_symbol, 'sell', greeks, otm_strike, 'put')
                #         orders.append(order_sell)
                #         order_buy = self.create_order(option_symbol, 'buy', greeks, deeper_otm_strike, 'put')
                #         orders.append(order_buy)

                elif market_signal == 'high_volatility':
                    # Straddle (buy both call and put at the same strike price)
                    # Buy both call and put
                    
                    if greeks['delta'] > 0.3 or greeks['delta'] < -0.3:
                      
                        order = self.create_order(timestamp, option_symbol, 'buy', option_premium, bid_size, ask_size)
                        orders.append(order)
                        order_len += 1

        orders_df = pd.DataFrame(orders)
        df_orders.to_csv('orders.csv', index=False)
        # print("Generated Orders:")
        # print(orders_df)
        return orders_df
   

    # def parse_data(self, options: pd.DataFrame) -> pd.DataFrame:
        
    #     df = options.copy()

    #     df.rename(columns={
    #         'ts_recv': 'timestamp',
    #         'bid_px_00': 'bid_price',
    #         'ask_px_00': 'ask_price',
    #         'bid_sz_00': 'bid_size',
    #         'ask_sz_00': 'ask_size',
    #         'symbol': 'option_symbol'
    #     }, inplace=True)

    #     df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    #     def parse_symbol(symbol: str):
    #         """
    #         EXAMPLE: SPX 20230120P2800000
    #         """
    #         try:
    #             numbers = symbol.split(" ")[1]
    #             date = numbers[:8]
    #             date_yymmdd = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
    #             action = numbers[8]
    #             strike_price = int(numbers[9:]) / 1000
    #             expiration_date = datetime.strptime(date_yymmdd, "%Y-%m-%d").date()
    #             option_type = 'Call' if action == 'C' else 'Put'
    #             return expiration_date, option_type, strike_price
    #         except (IndexError, ValueError):
    #             return None, None, None

    #     df[['expiration_date', 'option_type', 'strike_price']] = df['option_symbol'].apply(
    #         lambda x: pd.Series(parse_symbol(x))
    #     )
    #     df.dropna(subset=['expiration_date', 'option_type', 'strike_price'], inplace=True)

    #     df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
    #     df['spread'] = df['ask_price'] - df['bid_price']

    #     df['bid_price'] = df['bid_price'].astype(float)
    #     df['ask_price'] = df['ask_price'].astype(float)
    #     df['bid_size'] = df['bid_size'].astype(int)
    #     df['ask_size'] = df['ask_size'].astype(int)
    #     df['strike_price'] = df['strike_price'].astype(float)
        
    #     return df[[
    #         'timestamp',
    #         'instrument_id',
    #         'option_symbol',
    #         'expiration_date',
    #         'option_type',
    #         'strike_price',
    #         'bid_price',
    #         'ask_price',
    #         'bid_size',
    #         'ask_size',
    #         'mid_price',
    #         'spread'
            
    #     ]] 
    # Greek calculation functions
    def calculate_delta(self, S, K, T, r, sigma, option_type):
        """
        Calculates the Delta of an option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'Call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        return delta

    def calculate_theta(self, S, K, T, r, sigma, option_type):
        """
        Calculates the Theta of an option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        term1 = - (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
        if option_type == 'Call':
            theta = (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        return theta

    def calculate_gamma(self, S, K, T, r, sigma):
        """
        Calculates the Gamma of an option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    # 
    def calculate_vega(self, S, K, T, r, sigma):
        """
        Calculates the Vega of an option.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return vega

    def calculate_rho(self, S, K, T, r, sigma, option_type):
        
        # Calculates the Rho of an option.
        
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'Call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        return rho  
   
  
# st = Strategy()
# st.generate_orders()