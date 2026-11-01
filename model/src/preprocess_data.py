from model.src.data_download import get_data
import pandas as pd
import numpy as np
import sklearn as skl
import pickle


def remove_columns_with_none(df: pd.DataFrame) -> None:
    """
    Удаляет параметры с большим количеством пропусков.
    """
    for clm in df.columns:
        if np.sum(df[[clm]].isnull()) / df.shape[0] >= 0.35:
            df.drop(clm, axis=1, inplace=True)


def convert_str_to_num(df: pd.DataFrame) -> None:
    """
    Преобразует значения из строки  в числа.
    """
    for clm in ["общая площадь", "площадь кухни", "высота потолков", "жилая площадь"]:
        new_values = []
        for item in df[clm]:
            if isinstance(item, str):
                new_values.append(float(item.split()[0].replace(",", ".")))
            else:
                new_values.append(item)
        df[clm] = new_values

    new_values = []
    for item in df["стоимость"]:
        new_values.append(float("".join(item.split()[:-1])))

    df["стоимость"] = new_values


def transform_category_clmns(df: pd.DataFrame) -> None:
    """
    Преобразует параметры с категориальными значениями.
    """
    # 0 - 0.15 - нижние
    # 0.15 - 0.40 - ниже - среднего
    # 0.40 - 0.60 - средние
    # 0.60 - 0.85 - выше - среднего
    # 0.85 - 1 - высокие

    new_values = []
    for item in df["этаж"]:
        if isinstance(item, str):
            flat_class = int(item.split()[0]) / int(item.split()[2])
            if flat_class <= 0.15:
                new_values.append("нижние")
            elif 0.15 < flat_class <= 0.40:
                new_values.append("ниже - среднего")
            elif 0.40 < flat_class <= 0.60:
                new_values.append("средние")
            elif 0.60 < flat_class <= 0.85:
                new_values.append("выше - среднего")
            else:
                new_values.append("высокие")
        else:
            new_values.append(None)

    df["этаж"] = new_values

    # 1 санузел
    # 2 - 3 санузла
    # больше 3
    # Раздельный

    new_values_toilet_class = []
    new_values_sep_toilet = []
    for item in df["санузел"]:
        if isinstance(item, str):
            item_split = item.split()
            if len(item_split) == 2:
                toilet_amount = int(item_split[0])
            else:
                toilet_amount = int(item_split[0]) + int(item_split[2])
            if "раздельн" in item:
                new_values_sep_toilet.append(1)
            else:
                new_values_sep_toilet.append(0)
            if toilet_amount == 1:
                new_values_toilet_class.append("1_санузел")
            elif toilet_amount <= 3:
                new_values_toilet_class.append("2_3_санузла")
            else:
                new_values_toilet_class.append("больше 3 санузла")
        else:
            new_values_toilet_class.append(None)
            new_values_sep_toilet.append(None)
    df["санузел"] = new_values_toilet_class
    df["раздельный_санузел"] = new_values_sep_toilet

    list_lifts_amount = []

    for i in df["количество лифтов"]:
        if isinstance(i, str):
            item_split = i.split()
            if len(item_split) == 2:
                lifts_amount = int(item_split[0])
            else:
                lifts_amount = int(item_split[0]) + int(item_split[2])
            list_lifts_amount.append(lifts_amount)

    percent_1 = np.percentile(list_lifts_amount, 35)
    percent_2 = np.percentile(list_lifts_amount, 65)

    with open("model/src/cache_preprocess/percents.pkl", "wb") as f:
        pickle.dump([percent_1, percent_2], f)

    new_values_lift_class = []
    new_values_bages_lift = []
    for item in df["количество лифтов"]:
        if isinstance(item, str):
            item_split = item.split()
            if len(item_split) == 2:
                lifts_amount = int(item_split[0])
            else:
                lifts_amount = int(item_split[0]) + int(item_split[2])
            if "груз" in item:
                new_values_bages_lift.append(1)
            else:
                new_values_bages_lift.append(0)
            if lifts_amount <= percent_1:
                new_values_lift_class.append("немного_лифтов")
            elif percent_1 < lifts_amount <= percent_2:
                new_values_lift_class.append("среднее_кол_лифтов")
            else:
                new_values_lift_class.append("много_лифтов")
        else:
            new_values_lift_class.append(None)
            new_values_bages_lift.append(None)
    df["количество лифтов"] = new_values_lift_class
    df["грузовой_лифт"] = new_values_bages_lift


def fill_none(df: pd.DataFrame) -> None:
    """
    Заполняет пропуски в параметрах.
    """
    category_clmns = ["санузел", "количество лифтов", "тип дома", "раздельный_санузел", "грузовой_лифт"]
    contin_clmns = ["площадь кухни", "высота потолков", "жилая площадь", "долгота", "широта"]

    for clm in contin_clmns:
        mean_value = np.mean(df[clm])
        df[clm] = df[clm].fillna(mean_value)

    for clm in category_clmns:
        moda_value = df[clm].mode()[0]
        df[clm] = df[clm].fillna(moda_value)

    mean_coordinates = [np.mean(df["долгота"]), np.mean(df["широта"])]

    with open("model/src/cache_preprocess/latitude_longtitude.pkl", "wb") as f:
        pickle.dump(mean_coordinates, f)


def remove_outlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет выбросы в данных.
    """
    for clm_name in df.select_dtypes(include="number").columns[:-2]:
        Q1 = df[clm_name].quantile(0.25)
        Q3 = df[clm_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 4 * IQR
        upper_bound = Q3 + 4 * IQR

        df = df[df[clm_name] > lower_bound]
        df = df[df[clm_name] < upper_bound]

    df.reset_index(inplace=True, drop=True)

    return df


def encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Стандартизирует числовые параметры и кодирует категориальные.
    """
    category_clmns = ["этаж", "тип жилья", "санузел", "количество лифтов", "тип дома"]
    contin_clmns = ["площадь кухни", "высота потолков", "жилая площадь", "стоимость", "общая площадь", "долгота",
                    "широта"]

    one_hot_encoder = skl.preprocessing.OneHotEncoder(drop="first", sparse_output=False)
    std_scaler = skl.preprocessing.StandardScaler()

    std_scaler_for_price = skl.preprocessing.StandardScaler()
    std_scaler_for_predict = skl.preprocessing.StandardScaler()

    # for clm in contin_clmns:
    #     df[clm] = std_scaler.fit_transform(df[clm].to_numpy().reshape(-1, 1))
    std_scaler_for_price.fit(df[["стоимость"]].to_numpy())
    std_scaler_for_predict.fit(
        df[["площадь кухни", "высота потолков", "жилая площадь", "общая площадь", "долгота", "широта"]].to_numpy())
    std_scaler.fit(df[contin_clmns].to_numpy())
    df[contin_clmns] = std_scaler.transform(df[contin_clmns].to_numpy())

    # for clm in category_clmns:
    #     encoder_data = one_hot_encoder.fit_transform(df[clm].to_numpy().reshape(-1, 1)).toarray()
    #     df.drop(clm, axis=1, inplace=True)
    #     new_clmns_name = one_hot_encoder.get_feature_names_out([clm])
    #     encoder_df = pd.DataFrame(encoder_data, columns=new_clmns_name)
    #
    #     df = pd.concat([df, encoder_df], axis=1)
    one_hot_encoder.fit(df[category_clmns].to_numpy())
    encoder_data = one_hot_encoder.transform(df[category_clmns].to_numpy())
    df.drop(category_clmns, axis=1, inplace=True)
    new_clmns_name = one_hot_encoder.get_feature_names_out(input_features=category_clmns)

    encoder_df = pd.DataFrame(encoder_data, columns=new_clmns_name)
    df = pd.concat([df, encoder_df], axis=1)

    with open("model/src/cache_preprocess/one_hot_enc.pkl", "wb") as f:
        pickle.dump(one_hot_encoder, f)

    with open("model/src/cache_preprocess/std_scaler_for_price.pkl", "wb") as f:
        pickle.dump(std_scaler_for_price, f)

    with open("model/src/cache_preprocess/std_scaler_for_predict.pkl", "wb") as f:
        pickle.dump(std_scaler_for_predict, f)

    return df


def preprocess() -> None:
    """
    Запускает процесс предобработки данных.
    """
    df = get_data()

    remove_columns_with_none(df)
    convert_str_to_num(df)
    transform_category_clmns(df)
    fill_none(df)
    df = remove_outlines(df)
    df = encoding_features(df)

    df.to_csv("model/data/processed/prepared_data.csv", index_label=False)


def preprocess_for_predict(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает данные от клиента для предсказания стоимости квартиры.
    """

    flat_amount = user_df["этаж"][0]

    flat_class = int(flat_amount.split()[0]) / int(flat_amount.split()[2])

    if flat_class <= 0.15:
        flat_value = "нижние"
    elif 0.15 < flat_class <= 0.40:
        flat_value = "ниже - среднего"
    elif 0.40 < flat_class <= 0.60:
        flat_value = "средние"
    elif 0.60 < flat_class <= 0.85:
        flat_value = "выше - среднего"
    else:
        flat_value = "высокие"

    user_df["этаж"] = [flat_value]

    toilet_amount = user_df["санузел"][0]

    if toilet_amount == 1:
        toilet_value = "1_санузел"
    elif toilet_amount <= 3:
        toilet_value = "2_3_санузла"
    else:
        toilet_value = "больше 3 санузла"

    user_df["санузел"] = [toilet_value]

    with open("model/src/cache_preprocess/percents.pkl", "rb") as f:
        percents = pickle.load(f)

        lift_amount = user_df["количество лифтов"][0]

        if lift_amount <= percents[0]:
            user_df["количество лифтов"] = ["немного_лифтов"]
        elif percents[0] < lift_amount <= percents[1]:
            user_df["количество лифтов"] = ["среднее_кол_лифтов"]
        else:
            user_df["количество лифтов"] = ["много_лифтов"]

    category_clmns = ["этаж", "тип жилья", "санузел", "количество лифтов", "тип дома"]
    contin_clmns = ["площадь кухни", "высота потолков", "жилая площадь", "общая площадь", "долгота", "широта"]

    with open("model/src/cache_preprocess/std_scaler_for_predict.pkl", "rb") as f:
        std_scaler_for_predict = pickle.load(f)

        user_df[contin_clmns] = std_scaler_for_predict.transform(user_df[contin_clmns].to_numpy())

    with open("model/src/cache_preprocess/one_hot_enc.pkl", "rb") as f:
        one_hot_encoder = pickle.load(f)
        encoder_data = one_hot_encoder.transform(user_df[category_clmns].to_numpy())
        user_df.drop(category_clmns, axis=1, inplace=True)
        new_clmns_name = one_hot_encoder.get_feature_names_out(input_features=category_clmns)

        encoder_df = pd.DataFrame(encoder_data, columns=new_clmns_name)
        user_df = pd.concat([user_df, encoder_df], axis=1)

    return user_df
