import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.multivariate.pca import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

class PCA_TradingStrategy:
    def __init__(self, data):
        """
        Initialize the PCA Trading Strategy.

        Parameters:
        - data: pandas DataFrame containing option data with Greeks and prices.
        """
        self.data = data.copy()
        self.greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        self.scaler = StandardScaler()
        self.pca_model = None
        self.loadings = None
        self.scores = None
        self.threshold = None
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.initial_capital = 100000  

    def prepare_data(self):
        """
        Standardize the Greeks and perform PCA.
        """
        # Ensure the Greeks are available in the data
        for greek in self.greeks:
            if greek not in self.data.columns:
                raise ValueError(f"Missing Greek '{greek}' in the data.")

        # Standardize the Greeks
        X = self.data[self.greeks].values
        X_std = self.scaler.fit_transform(X)

        # Perform PCA using statsmodels
        self.pca_model = PCA(X_std, ncomp=len(self.greeks), method='svd', standardize=False)
        self.pca_model.fit()

        # Get loadings and scores
        self.loadings = self.pca_model.loadings  # Eigenvectors
        self.scores = self.pca_model.factors  # Principal component scores

        # Add the principal components to the data
        pc_columns = [f'PC{i+1}' for i in range(len(self.greeks))]
        self.data.reset_index(drop=True, inplace=True)
        scores_df = pd.DataFrame(self.scores, columns=pc_columns)
        self.data = pd.concat([self.data, scores_df], axis=1)

    def generate_signals(self, threshold):
        """
        Generate trading signals based on the principal components.

        Parameters:
        - threshold: float, the threshold for generating buy/sell signals.
        """
        self.threshold = threshold

        # Example: Use PC1 to generate signals
        self.data['signal'] = np.where(
            self.data['PC1'] > self.threshold, 'Buy',
            np.where(self.data['PC1'] < -self.threshold, 'Sell', 'Hold')
        )

    def backtest(self, stop_loss_pct, take_profit_pct):
        """
        Backtest the trading strategy.

        Parameters:
        - stop_loss_pct: float, the stop-loss percentage.
        - take_profit_pct: float, the take-profit percentage.

        Returns:
        - total_return: float, the total return of the strategy.
        - sharpe_ratio: float, the Sharpe ratio of the strategy.
        - max_drawdown: float, the maximum drawdown of the strategy.
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        capital = self.initial_capital
        position = 0  # 1 for long, -1 for short, 0 for no position
        entry_price = 0
        portfolio_values = [capital]
        positions = []
        returns = []

        # Ensure the data is sorted by timestamp
        self.data.sort_values(by='timestamp', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        for idx, row in self.data.iterrows():
            signal = row['signal']
            price = row['option_price']  # Use the appropriate price column
            date = row['timestamp']

            # Entry Logic
            if position == 0:
                if signal == 'Buy':
                    position = 1
                    entry_price = price
                    entry_date = date
                elif signal == 'Sell':
                    position = -1
                    entry_price = price
                    entry_date = date

            # Exit Logic
            elif position == 1:
                # Calculate return since entry
                return_since_entry = (price - entry_price) / entry_price
                # Check stop-loss or take-profit conditions
                if return_since_entry <= -self.stop_loss_pct or return_since_entry >= self.take_profit_pct or signal == 'Sell':
                    capital *= (1 + return_since_entry)
                    returns.append(return_since_entry)
                    position = 0
                    entry_price = 0
            elif position == -1:
                return_since_entry = (entry_price - price) / entry_price
                if return_since_entry <= -self.stop_loss_pct or return_since_entry >= self.take_profit_pct or signal == 'Buy':
                    capital *= (1 + return_since_entry)
                    returns.append(return_since_entry)
                    position = 0
                    entry_price = 0

            portfolio_values.append(capital)
            positions.append(position)

        # Add portfolio values and positions to data
        self.data['portfolio_value'] = portfolio_values[1:]
        self.data['position'] = positions

        # Calculate performance metrics
        total_return = (capital - self.initial_capital) / self.initial_capital

        # Calculate daily returns
        self.data['daily_return'] = self.data['portfolio_value'].pct_change().fillna(0)
        daily_returns = self.data['daily_return']

        # Sharpe Ratio
        if daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum Drawdown
        cumulative_returns = self.data['portfolio_value']
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        return total_return, sharpe_ratio, max_drawdown

    def optimize(self, thresholds, stop_losses, take_profits):
        """
        Optimize the strategy parameters to maximize the Sharpe ratio.

        Parameters:
        - thresholds: list of floats, thresholds to test for signal generation.
        - stop_losses: list of floats, stop-loss percentages to test.
        - take_profits: list of floats, take-profit percentages to test.

        Returns:
        - best_params: dict, the parameters that resulted in the best Sharpe ratio.
        - best_performance: dict, the performance metrics for the best parameters.
        """
        best_sharpe = -np.inf
        best_params = None
        best_performance = None

        for th in thresholds:
            self.generate_signals(threshold=th)
            for sl in stop_losses:
                for tp in take_profits:
                    total_return, sharpe_ratio, max_drawdown = self.backtest(stop_loss_pct=sl, take_profit_pct=tp)

                    # Check if this is the best Sharpe Ratio so far
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_params = {
                            'threshold': th,
                            'stop_loss_pct': sl,
                            'take_profit_pct': tp
                        }
                        best_performance = {
                            'total_return': total_return,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown
                        }

        # Set the best parameters
        if best_params:
            self.generate_signals(threshold=best_params['threshold'])
            self.backtest(
                stop_loss_pct=best_params['stop_loss_pct'],
                take_profit_pct=best_params['take_profit_pct']
            )

        return best_params, best_performance

    def get_loadings(self):
        """
        Get the PCA loadings (eigenvectors).

        Returns:
        - loadings_df: pandas DataFrame containing the loadings.
        """
        pc_columns = [f'PC{i+1}' for i in range(len(self.greeks))]
        loadings_df = pd.DataFrame(self.loadings, index=self.greeks, columns=pc_columns)
        return loadings_df

    def explained_variance(self):
        """
        Get the explained variance of each principal component.

        Returns:
        - explained_variance: pandas Series containing the explained variance ratio.
        """
        eigenvalues = self.pca_model.eigenvals
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        pc_columns = [f'PC{i+1}' for i in range(len(self.greeks))]
        explained_variance = pd.Series(explained_variance_ratio, index=pc_columns)
        return explained_variance
