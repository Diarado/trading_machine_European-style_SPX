import os
import pandas as pd
from datetime import datetime
import re

class Strategy:
  
  def __init__(self) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = datetime(2024, 1, 1)
    self.end_date : datetime = datetime(2024, 3, 30)
  
    self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

    self.underlying = pd.read_csv("data/underlying_data_hour.csv")
    self.underlying.columns = self.underlying.columns.str.lower()

  def load_or_parse_options(self, raw_file_path: str, parsed_file_path: str) -> pd.DataFrame:
        """
        Load parsed options data if available, otherwise parse raw data and save it.
        
        Parameters:
            raw_file_path (str): Path to the raw options data CSV file.
            parsed_file_path (str): Path to the file where parsed data should be stored.
        """
        if os.path.exists(parsed_file_path):
            print(f"Loading parsed options data from {parsed_file_path}")
            return pd.read_pickle(parsed_file_path)
        else:
            # Parse the raw data
            print(f"Parsing raw options data from {raw_file_path}")
            options = pd.read_csv(raw_file_path)
            parsed_options = self.parse_data(options)
            # Save the parsed data for future use
            parsed_options.to_pickle(parsed_file_path)
            return parsed_options
  
  def generate_orders(self) -> pd.DataFrame:
    # implement me!
    parsed_option = self.load_or_parse_options("data/cleaned_options_data.csv", "data/parsed_options_data.pkl")
    # pd.set_option('display.max_rows', 10)        
    # pd.set_option('display.max_columns', None)   
    # pd.set_option('display.width', None)        
    # pd.set_option('display.max_colwidth', None)   
    # print(parsed_option.head(10))
    pass
  



  # helper method
  def parse_data(self, options: pd.DataFrame) -> pd.DataFrame:
      """
      Parses the cleaned_options_data.csv and returns a cleaned DataFrame with columns:
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

      Parameters:
          options (pd.DataFrame): cleaned_options_data.csv

      Returns:
          pd.DataFrame: Parsed options data.
      """
      df = options.copy()

      df.rename(columns={
          'ts_recv': 'timestamp',
          'bid_px_00': 'bid_price',
          'ask_px_00': 'ask_price',
          'bid_sz_00': 'bid_size',
          'ask_sz_00': 'ask_size',
          'symbol': 'option_symbol'
      }, inplace=True)

      # convert timestamp to datetime
      # example: 2024-01-02T14:30:02.402838204Z
      df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')

      # From option_symbol, extract expiration date, option type, and strike price
      # Example: 'SPX   240119P04700000'
      # 
      # - '240119' -> Expiration Date: 2024-01-19
      # - 'P' -> Put
      # - '04700000' -> Strike Price: 4700.0000

      def parse_symbol(symbol: str):
          """
          Helper func
          Parses the option symbol to extract expiration date, option type, and strike price.
          Returns:
              tuple: (expiration_date, option_type, strike_price)
          """
          # first 6 digits is exp date
          # find either C or P
          # last 8 digits is strike price
          pattern = r'(\d{6})([CP])(\d{8})'
          match = re.search(pattern, symbol)
          if match:
              exp, type, strike = match.groups()
              # expiration date
              exp_date = datetime.strptime(exp, '%y%m%d').date()
              # option type
              option_type = 'Call' if type == 'C' else 'Put'
              # strike price
              strike_price = int(strike) / 1000  # last three digits are decimals
              return exp_date, option_type, strike_price
          else:
              print('bad input')
              return None, None, None

      # create 3 new columns from option symbol
      df[['expiration_date', 'option_type', 'strike_price']] = df['option_symbol'].apply(
          lambda x: pd.Series(parse_symbol(x))
      )
      df.dropna(subset=['expiration_date', 'option_type', 'strike_price'], inplace=True)

      # additional fields can be necessary
      # mid price
      df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2

      # spread
      df['spread'] = df['ask_price'] - df['bid_price']

      df['bid_price'] = df['bid_price'].astype(float)
      df['ask_price'] = df['ask_price'].astype(float)
      df['bid_size'] = df['bid_size'].astype(int)
      df['ask_size'] = df['ask_size'].astype(int)
      df['strike_price'] = df['strike_price'].astype(float)

      df = df[[
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

      return df

# st = Strategy()
# st.generate_orders()