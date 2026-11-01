import pandas as pd
import os


def get_data() -> pd.DataFrame:
    """
    Функция для получения "сырых" данных
    """
    # df = pd.read_csv("./information_about_flats_notebook_coordinates.csv")
    df = pd.read_csv("model/information_about_flats_with_coordinates.csv")

    return df
