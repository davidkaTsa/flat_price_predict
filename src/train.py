import sys
import os
cd = os.getcwd()
sys.path.append(f'{cd}')
import pandas as pd
import numpy as np
import joblib
import sklearn as skl
from sklearn.model_selection import train_test_split
from model.src.preprocess_data import preprocess
from models import lgbm

pd.set_option("display.max_columns", None)

preprocess()

df = pd.read_csv("model/data/processed/prepared_data.csv")

x_data = df.iloc[:, 1:]
y_data = df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

train_lgbm = lgbm.Dataset(x_train.to_numpy(), label=y_train.to_numpy())

learning_rate = np.linspace(0.05, 0.7, 3)
feature_frac = [0.7, 0.8]
num_iteration = np.linspace(100, 1000, 3, dtype=int)
best_model = None
best_score = 0

for l in learning_rate:
    for f in feature_frac:
        for n in num_iteration:
            param = {'objective': 'regression', "metric": ["l2"], "learning_rate": l, "feature_fraction": f,
                     "num_iterations": n, "verbosity": -1}
            model_lgbm = lgbm.train(params=param, train_set=train_lgbm)
            pred_lgbm = model_lgbm.predict(x_test)

            r2_lgbm = skl.metrics.r2_score(y_test, pred_lgbm)
            if r2_lgbm > best_score:
                best_score = r2_lgbm
                best_model = model_lgbm


joblib.dump(best_model, "model/models/model.pkl")
