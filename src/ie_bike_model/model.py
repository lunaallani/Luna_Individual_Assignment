import pandas as pd
import numpy as np
import os.path
from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "hour.csv")


def ffill_missing(ser):
    return ser.fillna(method="ffill")


def is_weekend(data):
    return data["dteday"].dt.day_name().isin(["Saturday", "Sunday"]).to_frame()


def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()


def train_and_persist():
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]

    ffiller = FunctionTransformer(ffill_missing)

    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )

    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )

    preprocessing = FeatureUnion(
        [
            ("is_weekend", FunctionTransformer(is_weekend)),
            ("year", FunctionTransformer(year)),
            ("column_transform", ct),
        ]
    )

    reg = Pipeline(
        [("preprocessing", preprocessing), ("model", RandomForestRegressor())]
    )

    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]

    reg.fit(X_train, y_train)
    dump(reg, "model.joblib")
    return "done"


def predict(dteday, hr, weathersit, temp, atemp, hum, windspeed):
    reg = load("model.joblib")

    model_data = {
        "dteday": [dteday],
        "hr": [hr],
        "weathersit": [weathersit],
        "temp": [temp],
        "atemp": [atemp],
        "hum": [hum],
        "windspeed": [windspeed],
    }

    df = pd.DataFrame(
        model_data,
        columns=["dteday", "hr", "weathersit", "temp", "atemp", "hum", "windspeed"],
    )

    # converting dteday into a datetime
    df["dteday"] = pd.to_datetime(df["dteday"])
    y_pred = reg.predict(df)

    return int(y_pred[0])
