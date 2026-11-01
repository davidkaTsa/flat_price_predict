import joblib
import lightgbm

def load_model()-> lightgbm:
    """
    Функция для загрузки сохраненной модели
    """
    return joblib.load("model/models/model.pkl")
