import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

import predict

np.random.seed(455)

dataset = pd.read_csv(
    "../../data/stock_history.csv", index_col="Date", parse_dates=["Date"]
).drop(["Dividends", "Stock Splits"], axis=1)
print(dataset.head())

print(dataset.describe())

print(dataset.isna().sum())

dataset = dataset[dataset["ticker"].isin(["MA"])]

tstart = 2013
tend = 2020


def train_test_plot(dataset, tstart, tend):
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{tend + 1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend + 1})", f"Test ({tend + 1} and beyond)"])
    plt.title("MasterCard stock price")
    plt.show()


train_test_plot(dataset, tstart, tend)


def train_test_split(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend + 1}":, "High"].values
    return train, test


training_set, test_set = train_test_split(dataset, tstart, tend)

sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps = 60
features = 1
# split into samples
X_train, y_train = split_sequence(training_set_scaled, n_steps)

# Reshaping X_train for model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

dataset_total = dataset.loc[:, "High"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
inputs = inputs.reshape(-1, 1)
# scaling
inputs = sc.transform(inputs)

# Split into samples
X_test, y_test = split_sequence(inputs, n_steps)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)


def create_lstm_model():
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer="RMSprop", loss="mse")

    print(lstm_model.summary())

    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)
    lstm_model.save("lstm_model.keras")
    return lstm_model


def create_gru_model():
    gru_model = Sequential()
    gru_model.add(GRU(units=125, activation="tanh", input_shape=(n_steps, features)))
    gru_model.add(Dense(units=1))
    gru_model.compile(optimizer="RMSprop", loss="mse")

    print(gru_model.summary())
    gru_model.fit(X_train, y_train, epochs=50, batch_size=32)
    gru_model.save("gru_model.keras")
    return gru_model


def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("MasterCard Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("MasterCard Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


def build_gru_model_and_predict():
    gru_model = create_gru_model()
    predicted_stock_price = gru_model.predict(X_test)
    print(f"predicted_stock_price before inverse transform = {predicted_stock_price}")

    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    plot_predictions(test_set, predicted_stock_price)
    return_rmse(test_set, predicted_stock_price)
    return predicted_stock_price


def build_model_and_predict(model_type: str):
    match model_type:
        case "LSTM":
            lstm_model = create_lstm_model()
            # prediction
            predicted_stock_price = lstm_model.predict(X_test)
            print(f"predicted_stock_price before inverse transform = {predicted_stock_price}")

            # inverse transform the values
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)

            plot_predictions(test_set, predicted_stock_price)
            return_rmse(test_set, predicted_stock_price)
        case "GRU":
            predicted_stock_price = build_gru_model_and_predict()
        case _:
            predicted_stock_price = build_gru_model_and_predict()

    return predicted_stock_price


NEEDS_TO_BUILD_ARG_INDEX = 1
MODEL_TYPE_ARG_INDEX = 2

if __name__ == "__main__":
    needs_to_build = sys.argv[NEEDS_TO_BUILD_ARG_INDEX]
    model_type = sys.argv[MODEL_TYPE_ARG_INDEX]

    if eval(needs_to_build.capitalize()):
        predicted_stock_price = build_model_and_predict(model_type.upper())
    else:
        data = np.array([
            164.36000061035156,
            166.50999450683594,
            166.47000122070312,
            167.64999389648438
        ])

        data = np.ndarray((1, 1), buffer=data, dtype=float)
        predicted_stock_price = predict.predict(data, model_type)
        return_rmse(data, predicted_stock_price)

        plt.plot(predicted_stock_price, color="red", label="Predicted")
        plt.title("MasterCard Stock Price Prediction")
        plt.xlabel("Days")
        plt.ylabel("Open price")
        plt.legend()
        plt.show()

    print(f"predicted_stock_price = {predicted_stock_price}")
