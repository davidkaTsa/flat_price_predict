import os
import sys
cd = os.getcwd()
sys.path.append(f'{cd}')
from fastapi import FastAPI
import numpy as np
import lightgbm
import pandas as pd
from model.src.preprocess_data import preprocess_for_predict
import pickle
from model.src import transform_address
from model.app.model_loader import load_model
from model.app.schema import ClientData

app = FastAPI(title="Flat Price Predict")

model = load_model()


@app.get("/")
def home() -> dict:
    """
    Функция для обработки get запроса
    """
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: ClientData) -> dict:
    """
    Функция для обработки post запроса.
    """
    features = list(data.model_dump().values())

    address = transform_address.transform_predict(features[-3])

    new_features = features[:-3]
    new_features.extend(address)
    new_features.extend(features[-2:])

    features = pd.DataFrame([new_features],
                            columns=["общая площадь", "площадь кухни", "этаж", "тип жилья", "высота потолков",
                                     "санузел", "количество лифтов", "тип дома", "жилая площадь", "долгота", "широта",
                                     "раздельный_санузел", "грузовой_лифт"])

    correct_data = preprocess_for_predict(features)
    pred_lgbm = model.predict(correct_data)[0]

    with open("model/src/cache_preprocess/std_scaler_for_price.pkl", "rb") as f:
        std_scaler_for_price = pickle.load(f)

        predicted_price = std_scaler_for_price.inverse_transform([[pred_lgbm]])[0][0]

    return {"predicted_price": int(predicted_price)}
