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

class Strategy:
  
    def __init__(self) -> None:
        self.capital : float = 100_000_000
        self.portfolio_value : float = 0

        self.start_date : datetime = datetime(2024, 1, 1)
        self.end_date : datetime = datetime(2024, 3, 30)
    
        self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv").copy()
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

        self.underlying = pd.read_csv("data/underlying_data_hour.csv").copy()
        self.underlying.columns = self.underlying.columns.str.lower()
        
        # Parse the 'date' column
        self.underlying['date'] = pd.to_datetime(self.underlying['date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
        self.underlying['date'] = self.underlying['date'].dt.tz_localize(None)
        
        columns_to_scale = ['open', 'high', 'low', 'close', 'adj close']
        self.underlying[columns_to_scale] = self.underlying[columns_to_scale] / 100
        
        self.spx_minute_data: pd.DataFrame = pd.read_csv("data/spx_minute_level_data_jan_mar_2024.csv").copy()
        self.spx_minute_data['price'] = self.spx_minute_data['price'] / 100
        # Convert the ms_of_day into a time format (milliseconds to timedelta)
        self.spx_minute_data['ms_of_day'] = pd.to_timedelta(self.spx_minute_data['ms_of_day'], unit='ms')
        # Convert ms_of_day from ET to UTC by adding 5 hours
        self.spx_minute_data['ms_of_day'] += pd.Timedelta(hours=5)
        # Convert the 'date' column to datetime format ('YYYYMMDD' format) in UTC
        self.spx_minute_data['date'] = pd.to_datetime(self.spx_minute_data['date'], format='%Y%m%d', utc=True)
        self.spx_minute_data['date'] = self.spx_minute_data['date'].dt.tz_localize(None)
        
        self.idx = 0

    def detect_market_signal(self, macd_value, macd_signal, avg_vega, days_to_expiration):
        if macd_value > macd_signal and avg_vega < 0.3:
            return 'bull'
        elif macd_value < macd_signal and avg_vega < 0.3:
            return 'bear'
        elif avg_vega >= 0.3:
            return 'high_volatility'
        elif days_to_expiration <= 3:
            return 'near_expiration'
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
    
    def create_order(self, option_id, action, greeks, strike_price, option_symbol):
 
        order_size = max(1, int(100 * abs(greeks['delta'])))  

        order = {
        'datetime': pd.Timestamp.now().isoformat(),  # Adjust datetime to match the example
        'option_symbol': option_symbol,
        'action': 'B' if action == 'buy' else 'S', 
        'order_size': order_size
        }
    
        return order

    def find_closest_strike(self, current_price, available_strikes):

        closest_strike = min(available_strikes, key=lambda strike: abs(strike - current_price))
        return closest_strike
    
    def generate_orders(self) -> pd.DataFrame:
        parsed_options = self.load_or_parse_options("data/cleaned_options_data.csv", "data/parsed_options_data.pkl")
        
        risk_free_rate = 0.03  # predetermined risk-free rate
        max_order_size = 100   # Adjustable maximum order size
        stop_loss_pct = 0.05   # Stop loss percentage
        take_profit_pct = 0.10 # Take profit percentage
        
        orders = []
        portfolio_value = self.capital
        daily_returns = []
        
        # Window for analyzing recent 26 price movements
        win_len = 26
        window = deque(self.spx_minute_data['price'].iloc[:27])
        
        cur_idx = 0  # Index pointer for parsed_options
        num_options = len(parsed_options)  # Total number of options
        
        for i, line in self.spx_minute_data.iloc[26:].iterrows():
            minute = line['ms_of_day']
            date = line['date']
            current_price = line['price']
            
            current_time = pd.Timestamp(date) + minute

            try:
                macd, signal = self.calculate_macd(window)
            except ValueError:
                print("MACD calculation error")
                continue
            
            options_today = []
            window.append(current_price)
            
            while len(window) > win_len:
                window.popleft()
            
            while cur_idx < num_options and parsed_options['timestamp'].iloc[cur_idx] < current_time:
                cur_idx += 1
                
            while cur_idx < num_options and parsed_options['timestamp'].iloc[cur_idx] < current_time + pd.Timedelta(milliseconds=6000):
                options_today.append(parsed_options.iloc[cur_idx]) 
                cur_idx += 1
            
            options_today = pd.DataFrame(options_today)
            greeks_list = []        
            
            for j, option in options_today.iterrows(): 
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
                    greeks_list.append(greeks)
                
                if not greeks_list:
                    print(f"No valid Greeks calculated for date {date}. Skipping this date.")
                    continue

                greeks_df = pd.DataFrame(greeks_list).set_index('option_symbol')
                print(greeks_df)

                # Extract the latest MACD and signal values
                macd_value = macd.iloc[-1]
                macd_signal = signal.iloc[-1]
                print(f"MACD Value: {macd_value}, Signal Value: {macd_signal}")

                # Implement the trading strategies based on the market signal
                avg_vega = greeks_df['vega'].mean()
                market_signal = self.detect_market_signal(macd_value, macd_signal, avg_vega, days_to_expiration)

                if market_signal == 'bull':
                    # Bull Call Spread (buy call with lower strike, sell call with higher strike)
                    for option_id, greeks in greeks_df.iterrows():
                        if greeks['delta'] > 0.3:
                            # Buy call with lower strike
                            order_buy = self.create_order(option_id, 'buy', greeks, greeks['strike_price'], 'call')
                            orders.append(order_buy)
                            # Sell call with higher strike
                            higher_strike = self.find_closest_strike(current_price + 2, options_today['strike_price'].unique())  # Arbitrary +2 strike difference
                            order_sell = self.create_order(option_id, 'sell', greeks, higher_strike, 'call')
                            orders.append(order_sell)

                elif market_signal == 'bear':
                    # Bear Put Spread (buy put with higher strike, sell put with lower strike)
                    for option_id, greeks in greeks_df.iterrows():
                        if greeks['delta'] < -0.3:
                            # Buy put with higher strike
                            order_buy = self.create_order(option_id, 'buy', greeks, greeks['strike_price'], 'put')
                            orders.append(order_buy)
                            # Sell put with lower strike
                            lower_strike = self.find_closest_strike(current_price - 2, options_today['strike_price'].unique())  # Arbitrary -2 strike difference
                            order_sell = self.create_order(option_id, 'sell', greeks, lower_strike, 'put')
                            orders.append(order_sell)

                elif market_signal == 'near_expiration':
                    # Iron Condor (sell OTM call and put, buy deeper OTM call and put)
                    for option_id, greeks in greeks_df.iterrows():
                        if greeks['delta'] > 0.3:
                        # Sell OTM call and buy deeper OTM call
                            otm_strike = self.find_closest_strike(current_price + 1, options_today['strike_price'].unique())
                            deeper_otm_strike = self.find_closest_strike(current_price + 2, options_today['strike_price'].unique())
                            order_sell = self.create_order(option_id, 'sell', greeks, otm_strike, 'call')
                            orders.append(order_sell)
                            order_buy = self.create_order(option_id, 'buy', greeks, deeper_otm_strike, 'call')
                            orders.append(order_buy)
                        elif greeks['delta'] < -0.3:
                            # Sell OTM put and buy deeper OTM put
                            otm_strike = self.find_closest_strike(current_price - 1, options_today['strike_price'].unique())
                            deeper_otm_strike = self.find_closest_strike(current_price - 2, options_today['strike_price'].unique())
                            order_sell = self.create_order(option_id, 'sell', greeks, otm_strike, 'put')
                            orders.append(order_sell)
                            order_buy = self.create_order(option_id, 'buy', greeks, deeper_otm_strike, 'put')
                            orders.append(order_buy)

                elif market_signal == 'high_volatility':
                    # Straddle (buy both call and put at the same strike price)
                    for option_id, greeks in greeks_df.iterrows():
                        # Buy both call and put
                        if greeks['delta'] > 0.3:
                            order_call = self.create_order(option_id, 'buy', greeks, greeks['strike_price'], 'call')
                            orders.append(order_call)
                        elif greeks['delta'] < -0.3:
                            order_put = self.create_order(option_id, 'buy', greeks, greeks['strike_price'], 'put')
                            orders.append(order_put)

        orders_df = pd.DataFrame(orders)
        print("Generated Orders:")
        print(orders_df)
        return orders_df
   

    def load_or_parse_options(self, raw_file_path: str, parsed_file_path: str) -> pd.DataFrame:
        if os.path.exists(parsed_file_path):
            print(f"Loading parsed options data from {parsed_file_path}")
            return pd.read_pickle(parsed_file_path)
        else:
            print(f"Parsing raw options data from {raw_file_path}")
            options = pd.read_csv(raw_file_path)
            parsed_options = self.parse_data(options)
            parsed_options.to_pickle(parsed_file_path)
            return parsed_options

    def parse_data(self, options: pd.DataFrame) -> pd.DataFrame:
        df = options.copy()

        df.rename(columns={
            'ts_recv': 'timestamp',
            'bid_px_00': 'bid_price',
            'ask_px_00': 'ask_price',
            'bid_sz_00': 'bid_size',
            'ask_sz_00': 'ask_size',
            'symbol': 'option_symbol'
        }, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')

        def parse_symbol(symbol: str):
            pattern = r'(\d{6})([CP])(\d{8})'
            match = re.search(pattern, symbol)
            if match:
                exp, type, strike = match.groups()
                exp_date = datetime.strptime(exp, '%y%m%d').date()
                option_type = 'Call' if type == 'C' else 'Put'
                strike_price = int(strike) / 100000
                return exp_date, option_type, strike_price
            else:
                return None, None, None

        df[['expiration_date', 'option_type', 'strike_price']] = df['option_symbol'].apply(
            lambda x: pd.Series(parse_symbol(x))
        )
        df.dropna(subset=['expiration_date', 'option_type', 'strike_price'], inplace=True)

        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['spread'] = df['ask_price'] - df['bid_price']

        df['bid_price'] = df['bid_price'].astype(float)
        df['ask_price'] = df['ask_price'].astype(float)
        df['bid_size'] = df['bid_size'].astype(int)
        df['ask_size'] = df['ask_size'].astype(int)
        df['strike_price'] = df['strike_price'].astype(float)
    
        return df[[
            'timestamp',
            'instrument_id',
            'option_symbol',
            'expiration_date',
            'option_type',
            'strike_price',
            'bid_price',
            'ask_price',
            'bid_size',
            'ask_size',
            'mid_price',
            'spread'
        ]] 
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
   
  
st = Strategy()
st.generate_orders()