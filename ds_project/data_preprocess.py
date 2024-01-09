import pandas as pd


def date_price_process(df):
  '''
  Функция обрабатывает регулярные и промо цены
  '''
  # преобразуем дату\цену
  df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
  df['Promo'] = df['Promo'].fillna(0)
  df.reset_index().set_index(['SKU_id', 'Store_id', 'Date'], inplace=True)
  df['Regular_Price'] = df['Regular_Price'].ffill().bfill()
  
  return df


def add_promo(df):
  '''
  Функция добавляет новые фичи, связанные с промо ценами
  '''
  df['Actual_Price'] = df['Promo_Price'].combine_first(df['Regular_Price'])
  df['Promo_percent'] = (1 - (df['Actual_Price'] / df['Regular_Price']))
  df = df.drop('Promo_Price', axis=1)

  df.reset_index(inplace=True)
  df['demand_expanding_mean'] = df.groupby(['Store_id', 'SKU_id'])['Demand'].expanding().mean().droplevel(['Store_id', 'SKU_id'])

  return df


def split_sku(data_lagged_features):
  '''
  Функция генерирует признаки выходных и праздников
  '''
  # преобразуем obj колонки в нужные
  data_lagged_features['weekday'] = data_lagged_features['weekday'].astype(int)
  data_lagged_features['monthday'] = data_lagged_features['monthday'].astype(int)
  data_lagged_features['is_weekend'] = data_lagged_features['is_weekend'].astype(int)

  return data_lagged_features