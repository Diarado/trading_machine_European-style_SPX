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
        
        # parse the 'date' column
        self.underlying['date'] = pd.to_datetime(self.underlying['date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
        self.underlying['date'] = self.underlying['date'].dt.tz_localize(None) 
        
        # print(self.underlying)
        columns_to_scale = ['open', 'high', 'low', 'close', 'adj close']
        self.underlying[columns_to_scale] = self.underlying[columns_to_scale] / 100
        
        self.spx_minute_data: pd.DataFrame = pd.read_csv("data/spx_minute_level_data_jan_mar_2024.csv").copy()

        # Convert the ms_of_day into a time format (milliseconds to timedelta)
        self.spx_minute_data['ms_of_day'] = pd.to_timedelta(self.spx_minute_data['ms_of_day'], unit='ms')
        # Convert the 'date' column to datetime format (assuming 'YYYYMMDD' format)
        self.spx_minute_data['date'] = pd.to_datetime(self.spx_minute_data['date'], format='%Y%m%d')
        # print(self.underlying)
        self.idx = 0
        
    def standardize(self, X):
        """
        Standardize the input data.
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def generate_trading_signals(self, pca_results, thresholds):
        """
        Generate trading signals based on PCA results and predefined thresholds.
        """
        signals = pd.DataFrame(index=range(pca_results.shape[0]))
        
        for i, threshold in thresholds.items():
            signals[f'PC{i+1}_signal'] = np.where(pca_results[:, i] > threshold, 1, 
                                                np.where(pca_results[:, i] < -threshold, -1, 0))
        
        return signals

    def calculate_order_size(self, signals, max_order_size, portfolio_value):
        """
        Calculate order size based on trading signals and portfolio value.
        """
        combined_signal = signals.sum(axis=1)
        order_size = (np.abs(combined_signal) / np.max(np.abs(combined_signal))) * (portfolio_value * 0.01)
        return np.minimum(order_size, max_order_size)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """
        Calculate the Sharpe ratio of the strategy.
        """
        excess_returns = returns - risk_free_rate / 252  # daily returns
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate the maximum drawdown of the strategy.
        """
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
        """
        Calculates the implied volatility using Brent's method.
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - T: Time to maturity (in years)
        - r: Risk-free interest rate
        - market_price: Observed market price of the option
        - option_type: 'Call' or 'Put'
        - plot: If True, plots the Brent's method process
        
        Returns:
        - Implied volatility
        """
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
            # Handle cases where brentq fails to find a root
            implied_vol = 1e-6
        return implied_vol

    ### current_price, option['strike_price'], T, risk_free_rate, implied_vol, option['option_type']
    def compute_option_greeks(self, S, K, T, r, sigma, option_type):
        """
        Computes all Greeks for a single option.
        Returns a dictionary of Greek values.
        """
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
    
    
    def generate_orders(self) -> pd.DataFrame:
        # Load or parse the options data
        parsed_options = self.load_or_parse_options("data/cleaned_options_data.csv", 
                                                    "data/parsed_options_data.pkl")
        
        risk_free_rate = 0.03  # predetermined risk-free rate
        max_order_size = 100   # Adjustable maximum order size
        stop_loss_pct = 0.05   # Stop loss percentage
        take_profit_pct = 0.10 # Take profit percentage
        
        orders = []
        portfolio_value = self.capital
        daily_returns = []
        
        # Window for analyzing recent 26 price movements
        win_len = 26
        window = deque()
        
        cur_idx = 0  # Index pointer for parsed_options
        num_options = len(parsed_options)  # Total number of options
        
        for i, line in self.spx_minute_data.iterrows():
            minute = line['ms_of_day']
            date = line['date']
            current_price = line['price']
            
            # Filtering options within the given time window for the current minute
            options_today = []
            while cur_idx < num_options and parsed_options['timestamp'].iloc[cur_idx] < minute + 6000:
                options_today.append(parsed_options.iloc[cur_idx]) 
                window.append(parsed_options['price'].iloc[cur_idx])
                while len(window) > 26:
                    window.popleft()
                cur_idx += 1
            
            # Example calculation of Greeks (use appropriate method for actual calculations)
            greeks_list = []        
            
            if i == 0: # skip the very first minute
                continue
            
            try:
                macd, signal_line = self.calculate_macd(window)
            except ValueError:
                print(f"MACD calculation error")
                continue  # Skip this date if MACD calculation fails
                    
            # starting from 2 minute
            for j, option in options_today.iterrows(): # options_today = options_tominute around 300 options
                # print('option: ')
                # print(option)
                 
                days_to_expiration = (option['expiration_date'] - date).days
                
                if days_to_expiration <= 0:
                    print(f"Option {option['instrument_id']} has expired. Skipping.")
                    continue  
                
                T = days_to_expiration / 365.0
                
                implied_vol = self.calculate_implied_volatility(current_price, option['strike_price'], 
                                                                T, risk_free_rate, option['mid_price'], 
                                                                option['option_type'])
                # print('cur_price: ' + str(current_price))
                # print('option strike_price' + str(option['strike_price']))
                # print('time: ' + str(T))
                # print('risk_rate: ' + str(risk_free_rate))
                # print('implied vol: ' + str(implied_vol))
                # print('option' + str(option['option_type']))
                greeks = self.compute_option_greeks(current_price, option['strike_price'], 
                                                    T, risk_free_rate, implied_vol, 
                                                    option['option_type'])
                # print(greeks)      
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

                # Implement MACD-based trading signals
                # Example:
                # If MACD crosses above the signal line, it's a bullish signal (buy)
                # If MACD crosses below the signal line, it's a bearish signal (sell)
                # For simplicity, we'll use the latest values to determine the signal
                if macd_value > macd_signal:
                    signal = 'buy'
                elif macd_value < macd_signal:
                    signal = 'sell'
                else:
                    signal = 'hold'
                
                print(f"MACD Signal for date {date}: {signal}")

                # Generate orders based on the signal and standardized Greeks
                for option_id, greeks in greeks_df.iterrows():
                    if signal == 'buy' and greeks['delta'] > 0.5:
                        quantity = min(max_order_size, int(portfolio_value // current_price))
                        if quantity > 0:
                            order = {
                                'datetime': option['timestamp'],
                                'option_id': option_id,
                                'action': 'B',
                                'quantity': quantity,
                            }
                            
                            #if date != and order_size <= int(row["ask_size"]) or order_size <= int(row["bid_size"]):
                            orders.append(order)
                            portfolio_value -= current_price * quantity
                            print(f"Generated BUY order: {order}")
                    
                    elif signal == 'sell' and greeks['delta'] < -0.5:
                        quantity = min(max_order_size, int(portfolio_value // current_price))
                        if quantity > 0:
                            order = {
                                'datetime': option['timestamp'],
                                'option_id': option_id,
                                'action': 'S',
                                'quantity': quantity,
                            }
                            orders.append(order)
                            portfolio_value += current_price * quantity
                            print(f"Generated SELL order: {order}")
                
                # Optionally, handle 'hold' signals or other strategies
            # Update daily_returns or other metrics as needed based on your strategy


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
   


'''
generate orders function (Todo: update the trading signals funcion)

def generate_orders(self) -> pd.DataFrame:
    parsed_options = self.load_or_parse_options("data/cleaned_options_data.csv", "data/parsed_options_data.pkl")
    
    risk_free_rate = 0.03  # predetermined
    max_order_size = 100  # Adjust as needed
    stop_loss_pct = 0.05  # Adjust as needed
    take_profit_pct = 0.10  # Adjust as needed

    orders = []
    portfolio_value = self.capital
    
    for date in parsed_options['timestamp'].dt.date.unique():
        options_today = parsed_options[parsed_options['timestamp'].dt.date == date]
        underlying_today = self.underlying[self.underlying['date'].dt.date == date]
        
        if len(underlying_today) == 0:
            continue
        
        current_prices = underlying_today[['date', 'adj close']].copy()
        current_prices['time'] = current_prices['date'].dt.time
        options_today = options_today.copy()
        options_today['time'] = options_today['timestamp'].dt.time
        
        date_list = current_prices['time'].tolist()

        greeks_list = []
        for _, option in options_today.iterrows():
            option_timestamp = option['time']
            idx = bisect.bisect_right(date_list, option_timestamp) - 1
            
            if idx >= 0:
                current_price = current_prices.iloc[idx]['adj close']
            else:
                continue

            # Find the closest available strike price
            available_strikes = options_today['strike_price'].unique()
            closest_strike = self.find_closest_strike(current_price, available_strikes)

            days_to_expiration = (option['expiration_date'] - date).days
            if days_to_expiration <= 0:
                continue
            
            T = days_to_expiration / 365.0
            implied_vol = self.calculate_implied_volatility(current_price, closest_strike, 
                                                            T, risk_free_rate, option['mid_price'], 
                                                            option['option_type'])
            greeks = self.compute_option_greeks(current_price, closest_strike, T, risk_free_rate, implied_vol, option['option_type'])
            
            if greeks:
                greeks['option_id'] = option['instrument_id']
                greeks_list.append(greeks)
        
        if not greeks_list:
            continue
        
        greeks_df = pd.DataFrame(greeks_list).set_index('option_id')

        # Placeholder for actual signal detection logic
        market_signal = self.detect_market_signal()  

        if market_signal == 'bull':
            # Bull Call Spread
            for option_id, greeks in greeks_df.iterrows():
                if greeks['delta'] > 0.3:  
                    # Buy call with lower strike, sell call with higher strike
                    order_buy = self.create_order(option_id, 'buy', greeks, closest_strike)
                    orders.append(order_buy)
                    higher_strike = self.find_closest_strike(current_price + 2, available_strikes)  # Arbitrary +2 strike difference
                    order_sell = self.create_order(option_id, 'sell', greeks, higher_strike)
                    orders.append(order_sell)

        elif market_signal == 'bear':
            # Bear Put Spread
            for option_id, greeks in greeks_df.iterrows():
                if greeks['delta'] < -0.3: 
                    # Buy put with higher strike, sell put with lower strike
                    order_buy = self.create_order(option_id, 'buy', greeks, closest_strike)
                    orders.append(order_buy)
                    lower_strike = self.find_closest_strike(current_price - 2, available_strikes)  # Arbitrary -2 strike difference
                    order_sell = self.create_order(option_id, 'sell', greeks, lower_strike)
                    orders.append(order_sell)

        elif market_signal == 'near_expiration':
            # Iron Condor (low volatility, near expiration)
            for option_id, greeks in greeks_df.iterrows():
                if greeks['delta'] > 0.3: 
                    # Sell OTM call and buy deeper OTM call
                    otm_strike = self.find_closest_strike(current_price + 1, available_strikes)
                    deeper_otm_strike = self.find_closest_strike(current_price + 2, available_strikes)
                    order_sell = self.create_order(option_id, 'sell', greeks, otm_strike)
                    orders.append(order_sell)
                    order_buy = self.create_order(option_id, 'buy', greeks, deeper_otm_strike)
                    orders.append(order_buy)
                elif greeks['delta'] < -0.3: 
                    # Sell OTM put and buy deeper OTM put
                    otm_strike = self.find_closest_strike(current_price - 1, available_strikes)
                    deeper_otm_strike = self.find_closest_strike(current_price - 2, available_strikes)
                    order_sell = self.create_order(option_id, 'sell', greeks, otm_strike)
                    orders.append(order_sell)
                    order_buy = self.create_order(option_id, 'buy', greeks, deeper_otm_strike)
                    orders.append(order_buy)

        elif market_signal == 'high_volatility':
            # Straddle (expecting significant market movement)
            for option_id, greeks in greeks_df.iterrows():
                # Buy both call and put
                if greeks['delta'] > 0.3:  
                    order_call = self.create_order(option_id, 'buy', greeks, closest_strike)
                    orders.append(order_call)
                elif greeks['delta'] < -0.3:  
                    order_put = self.create_order(option_id, 'buy', greeks, closest_strike)
                    orders.append(order_put)

        # Update portfolio value based on current orders and generated trades
        self.update_portfolio_value(orders)

    orders_df = pd.DataFrame(orders)
    return orders_df

'''
  
st = Strategy()
st.generate_orders()