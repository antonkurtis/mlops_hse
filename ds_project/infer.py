import os
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from omegaconf import DictConfig
from data_preprocess import *
from feature_generation import make_features, roll_agg_lag_features
from hydra import compose, initialize
import pandas as pd


def main(cfg: DictConfig):
    '''
    Функция реализует полный пайплайн предсказания модели
    с предобработкой данных, генерацией фичей, инференсом модели и
    выгрузкой результата в .csv файл
    '''
    # инициализируем конфиги
    initialize(version_base=None, config_path="../config")
    
    project_cfg = compose(config_name="project_config.yaml")
    
    # загружаем данные
    df_train = pd.read_csv(project_cfg["path"]["demand_train"])
    df_test = pd.read_csv(project_cfg["path"]["demand_test"])
    sub = pd.read_csv(project_cfg["path"]["sub_sample"])

    df = pd.concat([df_train, df_test.drop('id', axis=1)], axis=0, ignore_index=True)

    # предобработка данных
    df = date_price_process(df)
    df = add_promo(df)
    df.drop('index', axis=1, inplace=True)

    # генерация фичей
    data_lagged_features = roll_agg_lag_features(
        df, project_cfg["feature_generation"]["lags"], 
        project_cfg["feature_generation"]["rolls"]
        )
    
    # определяем границу трейна-предикта, используем дату как индекс
    data_lagged_features.set_index(project_cfg["constant"]["date_col"], inplace = True)

    # делим датасет на 2 по SKU (отключено ввиду упрощения, просто добавляем пару новых)
    data_lagged_features = split_sku(data_lagged_features)

    sd = '2016-05-23'  
    ed = '2016-06-19' 
    dates = pd.date_range(sd, ed)

    test_data = data_lagged_features[data_lagged_features.index.isin(dates)].drop(project_cfg["model"]["drop_columns"], axis=1)

    # загрузка модели
    model = CatBoostRegressor()

    model.load_model(os.path.join(project_cfg["path"]["models"], "catboost.cbm"))

    prediction = model.predict(test_data)
    
    sub['Demand'] = prediction
    sub.to_csv(        
        os.path.join(project_cfg["path"]["predictions"], "prediction.csv"),
        index=False,                
        )


if __name__ == "__main__":
    main()