import pandas as pd

def make_features(data, max_lag, rolling_mean_size):
  # генерим фичи
  # создаем лаги\агрегации
  for lag in range(1, max_lag + 1):
    data['lag_{}'.format(lag)] = data['Demand'].shift(lag)

  data['rolling_mean_{}'.format(lag)] = data['Demand'].shift().rolling(rolling_mean_size).mean()
  data['rolling_median_{}'.format(lag)] = data['Demand'].shift().rolling(rolling_mean_size).median()


def roll_agg_lag_features(new_data, lags, rolls):
  # создаем фичи из даты
  new_data["weekday"] = pd.to_numeric(new_data.Date.dt.weekday)
  new_data["monthday"] = pd.to_numeric(new_data.Date.dt.day)
  new_data['is_weekend'] = pd.to_numeric(new_data.weekday.isin([5,6])*1)

  # поэтапно применяем генерацию фичей
  make_features(new_data, lags[0], rolls[0])
  make_features(new_data, lags[1], rolls[1])
  make_features(new_data, lags[2], rolls[2])
  make_features(new_data, lags[3], rolls[3])
  make_features(new_data, lags[4], rolls[4])

  result = pd.DataFrame(columns = new_data.columns)

  for i in new_data['Store_id'].unique():
    frames = new_data[new_data['Store_id']==i]#[41:]
    result = pd.concat([result, frames])
  
  return result.reset_index(drop=True)