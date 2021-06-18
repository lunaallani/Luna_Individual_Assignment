import pytest

from ie_bike_model.model import train_and_persist, predict


def test_train_and_persist():

    model = train_and_persist()

    assert model == "done"


def test_predict():

    model = predict(
        dteday="2012-11-01",
        hr=10,
        weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy",
        temp=0.3,
        atemp=0.31,
        hum=0.8,
        windspeed=0.0,
    )
    assert model > 0
