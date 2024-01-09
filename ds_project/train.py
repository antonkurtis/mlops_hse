import os
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from data_preprocess import *
from feature_generation import make_features, roll_agg_lag_features
from hydra import compose, initialize
import metric
import mlflow
import pandas as pd


def main():
    # инициализируем конфиги
    initialize(version_base=None, config_path="../configs")
    
    project_cfg = compose(config_name="project_config.yaml")
    mlflow_cfg = compose(config_name="mlflow_config.yaml")
    catboost_cfg = compose(config_name="catboost_config.yaml")
    
    # загружаем данные
    df_train = pd.read_csv(project_cfg["path"]["demand_train"])
    df_test = pd.read_csv(project_cfg["path"]["demand_test"])
    df = pd.concat([df_train, df_test.drop('id', axis=1)], axis=0, ignore_index=True)

    # предобработка данных
    df = date_price_process(df)
    df = add_promo(df)
    df.drop('index', axis=1, inplace=True)

    # генерация фичей
    data_lagged_features = roll_agg_lag_features(df, project_cfg["feature_generation"]["lags"], project_cfg["feature_generation"]["rolls"])
    
    # определяем границу трейна-предикта, используем дату как индекс
    data_lagged_features.set_index(project_cfg["constant"]["date_col"], inplace = True)

    # делим датасет на 2 по SKU (отключено ввиду упрощения, просто добавляем пару новых)
    data_lagged_features = split_sku(data_lagged_features)

    # инициализируем модель
    model = CatBoostRegressor(**catboost_cfg["model_params"])

    sd = '2015-01-01'  
    ed = '2016-05-22' 
    dates = pd.date_range(sd, ed)

    train_data = data_lagged_features[data_lagged_features.index.isin(dates)].drop(project_cfg["model"]["drop_columns"], axis=1)
    y = data_lagged_features[data_lagged_features.index.isin(dates)].Demand

    # тренируем модель
    model.fit(
        X = train_data,           
        y = y         
        )
    
    model.save_model(os.path.join(project_cfg["path"]["models"], "catboost.cbm"), format="cbm")

    # mlflow и логирование
    if mlflow_cfg["mlflow"]["logging"]:
        mlflow.set_tracking_uri(uri=mlflow_cfg["mlflow"]["logging_uri"])
        mlflow.set_experiment(mlflow_cfg["mlflow"]["experiment_name"])

        with mlflow.start_run():
            mlflow.log_params(mlflow_cfg["model_params"])

            metrics = metric.Metrics(
                actual=y, 
                prediction=model.predict(train_data)
            )

        mlflow.log_metric("WAPE", metrics.wape())
        mlflow.log_metric("MAE", metrics.mae())

        mlflow.set_tag(mlflow_cfg["mlflow"]["tag_name"], mlflow_cfg["mlflow"]["tag_value"])

        signature = infer_signature(
            train_data,                                      
            model.predict(train_data)
            )
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=mlflow_cfg["mlflow"]["artifact_path"],
            signature=signature,
            input_example=train_data,
            registered_model_name=mlflow_cfg["mlflow"]["registered_model_name"],
        )


if __name__ == "__main__":
    main()