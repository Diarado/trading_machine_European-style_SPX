import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt

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

    
    # 
    def black_scholes_call(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    

    def visualize_pca_results(self, pca_results, explained_variance_ratio):
        plt.figure(figsize=(12, 5))
        
        # Plot PCA results
        plt.subplot(1, 2, 1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Results')
        
        # Plot explained variance ratio
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')
        
        plt.tight_layout()
        plt.show()
    

    def calculate_implied_volatility(self, S, K, T, r, market_price, option_type):
        def objective(sigma):
            if option_type == 'Call':
                return self.black_scholes_call(S, K, T, r, sigma) - market_price
            else:
                return self.black_scholes_put(S, K, T, r, sigma) - market_price
        
        try:
            return brentq(objective, 1e-6, 10)  # volatility is between 0.0001% and 1000%
        except ValueError:
            return np.nan  # Return NaN if unable to find a root

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

    def perform_pca(self, X, n_components=None):
        """
        Perform PCA on the input data.
        
        :param X: Input data (standardized)
        :param n_components: Number of components to keep
        :return: PCA results and explained variance ratios
        """
        # Remove any rows with NaN or inf values
        X = X[~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)]
        
        if X.shape[0] == 0:
            print("No valid data for PCA after removing NaN and inf values.")
            return None, None

        # Compute the covariance matrix
        cov_matrix = np.cov(X.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute explained variance ratios
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        # Select number of components
        if n_components is None:
            n_components = X.shape[1]
        
        # Project data onto principal components
        pca_results = np.dot(X, eigenvectors[:, :n_components])
        
        return pca_results, explained_variance_ratio[:n_components]

    def generate_orders(self) -> pd.DataFrame:
        parsed_options = self.load_or_parse_options("data/cleaned_options_data.csv", "data/parsed_options_data.pkl")
        
        risk_free_rate = 0.03 # predetermined
        max_order_size = 100 # TODO: to be adjusted
        stop_loss_pct = 0.05 # TODO.
        take_profit_pct = 0.10 # TODO.
        
        orders = []
        portfolio_value = self.capital
        daily_returns = []
        
        for date in parsed_options['timestamp'].dt.date.unique():
            options_today = parsed_options[parsed_options['timestamp'].dt.date == date]
            underlying_today = self.underlying[self.underlying['date'].dt.date == date]
            print(underlying_today)
            if len(underlying_today) == 0:
                continue
            
            current_price = underlying_today['adj close'].iloc[-1]
            
            print('cur_price: ' + str(current_price))
            # Calculate Greeks for all options
            greeks_list = []
            pd.set_option('display.max_columns', None)
            print(options_today)
            for _, option in options_today.iterrows():
                days_to_expiration = (option['expiration_date'] - date).days
                T = days_to_expiration / 365.0
                
                implied_vol = self.calculate_implied_volatility(current_price, option['strike_price'], T, risk_free_rate, option['mid_price'], option['option_type'])
                print('cur_price: ' + str(current_price))
                print('option strike_price' + str(option['strike_price']))
                print('time: ' + str(T))
                print('risk_rate: ' + str(risk_free_rate))
                print('implied vol: ' + str(implied_vol))
                print('option' + str(option['option_type']))
                greeks = self.compute_option_greeks(current_price, option['strike_price'], T, risk_free_rate, implied_vol, option['option_type'])
                # print(greeks)
                if greeks:
                    greeks['option_id'] = option['instrument_id']
                    greeks_list.append(greeks)
            
            if not greeks_list:
                print(f"No valid Greeks calculated for date {date}. Skipping this date.")
                continue

            greeks_df = pd.DataFrame(greeks_list).set_index('option_id')
            
            # Perform PCA on standardized Greeks
            standardized_greeks = self.standardize(greeks_df.values)
            pca_results, explained_variance_ratio = self.perform_pca(standardized_greeks, n_components=2)
            
            if pca_results is None:
                print(f"PCA failed for date {date}. Skipping this date.")
                continue

            self.visualize_pca_results(pca_results, explained_variance_ratio)
            
            # Generate trading signals
            thresholds = {0: 1.0, 1: 0.5}  # Example thresholds, adjust as needed
            signals = self.generate_trading_signals(pca_results, thresholds)
            
            # Calculate order sizes
            order_sizes = self.calculate_order_size(signals, max_order_size, portfolio_value)
            
            # Generate orders based on signals and order sizes
            daily_pnl = 0
            for i, (option_id, signal) in enumerate(signals.iterrows()):
                if signal.sum() != 0:
                    option = options_today[options_today['instrument_id'] == greeks_df.index[i]].iloc[0]
                    order_size = int(order_sizes[i])
                    action = 'B' if signal.sum() > 0 else 'S'
                    
                    orders.append({
                        'datetime': option['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'option_symbol': f"SPX   {option['expiration_date'].strftime('%y%m%d')}{'C' if option['option_type'] == 'Call' else 'P'}{int(option['strike_price']*1000):08d}",
                        'action': action,
                        'order_size': order_size,
                        'stop_loss': option['mid_price'] * (1 - stop_loss_pct) if action == 'B' else option['mid_price'] * (1 + stop_loss_pct),
                        'take_profit': option['mid_price'] * (1 + take_profit_pct) if action == 'B' else option['mid_price'] * (1 - take_profit_pct)
                    })
                    
                    # Simplified P&L calculation ( a more complex model)
                    daily_pnl += order_size * (option['mid_price'] * take_profit_pct if action == 'B' else option['mid_price'] * stop_loss_pct)
            
            # Update portfolio value and calculate daily return
            portfolio_value += daily_pnl
            daily_returns.append(daily_pnl / portfolio_value)
        
        result = pd.DataFrame(orders)
        print(f"Final number of orders generated: {len(result)}")
        print(result.head(10))
        
        # Calculate performance metrics
        sharpe_ratio = self.calculate_sharpe_ratio(np.array(daily_returns), risk_free_rate)
        max_drawdown = self.calculate_max_drawdown(pd.Series(daily_returns).cumsum())
        
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Max Drawdown: {max_drawdown}")
        
        return result

   

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
        print('delta: ' + str(delta))
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