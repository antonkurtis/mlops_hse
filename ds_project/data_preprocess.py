import pandas as pd


def date_price_process(df):
  # преобразуем дату\цену
  df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
  # df.set_index(['Date'], inplace=True)
  df['Promo'] = df['Promo'].fillna(0)
  df.reset_index().set_index(['SKU_id', 'Store_id', 'Date'], inplace=True)
  df['Regular_Price'] = df['Regular_Price'].ffill().bfill()
  # df.reset_index().set_index(['Date'], inplace=True)
  
  return df

  # переписываем финальную цену и пара новых фичей
def add_promo(df):
  df['Actual_Price'] = df['Promo_Price'].combine_first(df['Regular_Price'])
  df['Promo_percent'] = (1 - (df['Actual_Price'] / df['Regular_Price']))
  df = df.drop('Promo_Price', axis=1)

  df.reset_index(inplace=True)
  df['demand_expanding_mean'] = df.groupby(['Store_id', 'SKU_id'])['Demand'].expanding().mean().droplevel(['Store_id', 'SKU_id'])

  return df


def split_sku(data_lagged_features):
  # преобразуем obj колонки в нужные
  data_lagged_features['weekday'] = data_lagged_features['weekday'].astype(int)
  data_lagged_features['monthday'] = data_lagged_features['monthday'].astype(int)
  data_lagged_features['is_weekend'] = data_lagged_features['is_weekend'].astype(int)

  # делим датасет по sku (отключено ввиду упрощения)
  # sku1_train = data_lagged_features[data_lagged_features['SKU_id'] == 1]
  # sku2_train = data_lagged_features[data_lagged_features['SKU_id'] == 2]

  return data_lagged_features