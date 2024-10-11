import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re

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
        self.underlying['date'] = self.underlying['date'].dt.tz_localize(None)  # Remove timezone info

    def generate_orders(self) -> pd.DataFrame:
        parsed_options = self.load_or_parse_options("data/cleaned_options_data.csv", "data/parsed_options_data.pkl")
        
        lookback_period = 10  # lookback period for trend analysis
        volatility_threshold = 0.001  # volatility threshold
        max_days_to_expiration = 30  # max days to expiration for selected options
        
        orders = []
        
        for date in parsed_options['timestamp'].dt.date.unique():
            options_today = parsed_options[parsed_options['timestamp'].dt.date == date]
            underlying_today = self.underlying[self.underlying['date'].dt.date == date]
            
            if len(underlying_today) == 0:
                continue
            
            current_price = underlying_today['close'].iloc[-1]
            
            # trend using exponential moving average
            if len(self.underlying) >= lookback_period:
                ema = self.underlying['close'].ewm(span=lookback_period, adjust=False).mean()
                trend = (current_price - ema.iloc[-1]) / ema.iloc[-1]
            else:
                trend = 0
            
            # recent volatility
            if len(self.underlying) >= lookback_period:
                recent_returns = self.underlying['close'].pct_change().tail(lookback_period)
                volatility = recent_returns.std()
            else:
                volatility = 0
            
            # only keep options with expiration date <= max
            max_expiration = date + timedelta(days = max_days_to_expiration)
            valid_options = options_today[options_today['expiration_date'] <= max_expiration]
            
            # orders based on market conditions
            if trend > 0.001 and volatility < volatility_threshold:
                calls = valid_options[(valid_options['option_type'] == 'Call') & 
                                      (valid_options['strike_price'] > current_price) & 
                                      (valid_options['strike_price'] < current_price * 1.05)]
                if not calls.empty:
                    best_call = calls.loc[calls['mid_price'].idxmin()]
                    order_size = int(min(100, max(10, abs(trend) * 1000)))  # more order if trend is high
                    orders.append({
                        'datetime': best_call['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'option_symbol': f"SPX   {best_call['expiration_date'].strftime('%y%m%d')}C{int(best_call['strike_price']*1000):08d}",
                        'action': 'B',
                        'order_size': order_size
                    })
            elif trend < -0.001 and volatility < volatility_threshold:
                puts = valid_options[(valid_options['option_type'] == 'Put') & 
                                     (valid_options['strike_price'] < current_price) & 
                                     (valid_options['strike_price'] > current_price * 0.95)]
                if not puts.empty:
                    best_put = puts.loc[puts['mid_price'].idxmin()]
                    order_size = int(min(100, max(10, abs(trend) * 1000)))  # Scale order size with trend
                    orders.append({
                        'datetime': best_put['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'option_symbol': f"SPX   {best_put['expiration_date'].strftime('%y%m%d')}P{int(best_put['strike_price']*1000):08d}",
                        'action': 'B',
                        'order_size': order_size
                    })
            elif volatility >= volatility_threshold:
                atm_options = valid_options[abs(valid_options['strike_price'] - current_price) < 5]
                if not atm_options.empty:
                    atm_call = atm_options[atm_options['option_type'] == 'Call'].iloc[0]
                    atm_put = atm_options[atm_options['option_type'] == 'Put'].iloc[0]
                    order_size = int(min(50, max(5, volatility * 1000)))  # Scale order size with volatility
                    orders.extend([
                        {
                            'datetime': atm_call['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                            'option_symbol': f"SPX   {atm_call['expiration_date'].strftime('%y%m%d')}C{int(atm_call['strike_price']*1000):08d}",
                            'action': 'S',
                            'order_size': order_size
                        },
                        {
                            'datetime': atm_put['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                            'option_symbol': f"SPX   {atm_put['expiration_date'].strftime('%y%m%d')}P{int(atm_put['strike_price']*1000):08d}",
                            'action': 'S',
                            'order_size': order_size
                        }
                    ])
        
        result = pd.DataFrame(orders)
        print(f"Final number of orders generated: {len(result)}")
        print(result.head(10))
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
                strike_price = int(strike) / 1000
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

# st = Strategy()
# st.generate_orders()
# li edit
st = Strategy()