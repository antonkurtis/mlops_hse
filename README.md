## Эпилог
В данном проекте реализован упрощенный вариант прогнозирования временных рядов из финального задания на kaggle курса  TS Forecasting  МОВС (https://www.kaggle.com/competitions/atsf-winter23-hw3/overview).

Стоит отметить, что в проект завернут сильно упрощенный вариант, ибо лучшее решение на private (за авторством моей скромной персоны) требует очень хитрого обучения множественных моделей с не менее хитрым сиеминутным инференсом. Мне показалось не совсем уместным сохранение более чем 100 моделей и дикой грязью в mlflow. Поэтому принято волевое решение свести все к одной модели (но в ноутбуке есть топовое решение)

## Создаем и настраиваем окружение, клонируем репозиторий
```
git clone https://github.com/antonkurtis/mlops_hse.git

cd mlops_hse

conda create -n mlops python=3.9 -y -q        

conda activate mlops

conda install poetry -y

poetry install                                                                                      
```

## Тянем данные с Google Drive
```
dvc pull 
```

## Запуск предсказаний предобученной моделью
```
poetry run python ds_project/infer.py
```

## Дообучение загруженной модели и логирование mlflow.
```
mlflow server --host 127.0.0.1 --port 8080                   

poetry run python ds_project/train.py                
```

## Добавление новых данных в dvc
```
dvc add data/{file_name}.csv

dvc add models/{new_model_name}.cbm 

dvc remote add --default myremote gdrive://{folder_id}

dvc remote modify myremote gdrive_acknowledge_abuse true

dvc push
```

## Описание работы
В результате работы train.py и infer.py происходит:
- загрузка и предобработка данных
- генерация новых фичей
- обучение\предсказание модели
- логирование эксперимента


`feature_generation.py`

Скрипт генерирует лаговые, агрегационные признаки



`data_preprocess.py`

Скрипт преобразует сырые признаки, такие как "Дата", "Спрос" во что то более осознанное,
например, бинарные признаки "Выходные" и "Праздники" а так же аггрегации по спросу


В роли модели выступает  CatBoostRegressor.
Все гиперпараметры модели, а так же полученные метрики (искренне прошу не смотреть на них, ибо причины описаны в самом начале), 
артeфакты и тп можно посмотреть в логах эксперимента mlflow
