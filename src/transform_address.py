import geopy.exc
import pandas as pd
from geopy.geocoders import Nominatim
import pickle
import time
import datetime


def transform_predict(address: str) -> list:
    """
    Функция для преобразования адреса в широту и долготу. Используется при получении запроса от клиента.
    """
    try:
        geolocator = Nominatim(user_agent="flat_app")
        location = geolocator.geocode(address)
    except Exception:
        location = None
    if location is not None:
        return [location.longitude, location.latitude]
    else:
        with open("model/src/cache_preprocess/latitude_longtitude.pkl", "rb") as f:
            mean_coordinates = pickle.load(f)

        return mean_coordinates


def transform_analysis(df: pd.DataFrame) -> None:
    """
    Функция для преобразования адреса в широту и долготу в сырых данных.
    """
    geolocator = Nominatim(user_agent="flat_app")
    # locat_longitude = []
    # locat_latitide = []
    i = 40000
    with open("position.pkl", "rb") as f:
        position_loaded = pickle.load(f)
        locat_longitude = position_loaded[0]
        locat_latitide = position_loaded[1]


    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    while i != len(df["адрес"]):
        try:
            address = df["адрес"][i]
            address_details = address.split(",")
            location = geolocator.geocode(f"москва, {address_details[-2]}, {address_details[-1]}")
            if location is not None:
                locat_latitide.append(location.latitude)
                locat_longitude.append(location.longitude)
            else:
                locat_latitide.append(None)
                locat_longitude.append(None)
            if i % 500 == 0:
                print(i, datetime.datetime.now())
            i += 1
            time.sleep(1)
        except geopy.exc.GeocoderUnavailable as e:
            time.sleep(1)
        except IndexError as e:
            locat_latitide.append(None)
            locat_longitude.append(None)
            i += 1
            time.sleep(1)

    with open("position.pkl", "wb") as f:
        pickle.dump([locat_longitude, locat_latitide], f)

    df.insert(len(df.columns), "долгота", locat_longitude)
    df.insert(len(df.columns), "широта", locat_latitide)

    df.drop("адрес", axis=1, inplace=True)
    df.to_csv("information_about_flats_with_coordinates.csv", index_label=False)

