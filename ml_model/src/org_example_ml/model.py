"""
Predict the stock price using LSTM or GRU model
"""

import dataclasses
import logging
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers import Dropout, GRU, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# noinspection PyUnresolvedReferences
import org_example_ml.logging_config  # pylint: disable=unused-import
from org_example_ml import app_config
from org_example_ml import predict


@dataclasses.dataclass
class ModelTrainingContext:  # pylint: disable=too-many-instance-attributes
    """
    Holds metadata required for building and training a model
    """

    model_type: str
    scaler: MinMaxScaler
    x_train_stock_price_dataset: np.ndarray
    y_train_stock_price_dataset: np.ndarray
    input_shape: [int, int]
    x_test_stock_price_dataset: np.ndarray
    y_test_stock_price_dataset: np.ndarray
    model_output_file_path: str


def load_dataset():
    """
    Loads dataset and then apply filter keeping only MA tickers
    """
    stock_price_dataset = pd.read_csv(
        app_config.DATASET_PATH, index_col="Date", parse_dates=["Date"]
    ).drop(["Dividends", "Stock Splits"], axis=1)
    logging.info(stock_price_dataset.head())
    logging.info(stock_price_dataset.describe())
    logging.info(stock_price_dataset.isna().sum())
    return stock_price_dataset[stock_price_dataset["ticker"].isin(["MA"])]


def plot_train_test(train_dataset: pd.DataFrame, year_start: int, year_end: int):
    """
    Plot dataset for given time period.
    """
    train_dataset.loc[f"{year_start}":f"{year_end}", "High"].plot(figsize=(16, 4), legend=True)
    train_dataset.loc[f"{year_end + 1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {year_end + 1})", f"Test ({year_end + 1} and beyond)"])
    plt.title("Stock price")
    plt.show()


def train_test_split(stock_price_dataset: pd.DataFrame, year_start: int, year_end: int):
    """
    Split dataset to train and set according to specified time period
    """
    train = stock_price_dataset.loc[f"{year_start}":f"{year_end}", "High"].values
    test = stock_price_dataset.loc[f"{year_end + 1}":, "High"].values
    return train, test


def split_sequence(sequence, number_of_steps: int):
    """
    Split into samples
    """
    x, y = [], []
    for i in range(len(sequence)):
        end_index = i + number_of_steps
        if end_index > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_index], sequence[end_index]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def train_model(
    training_context: ModelTrainingContext,
    sequential_model: Sequential,
    callbacks: list[Callback] = None,
) -> History:
    """
    Train the model and then save it to file
    """
    history = sequential_model.fit(
        x=training_context.x_train_stock_price_dataset,
        y=training_context.y_train_stock_price_dataset,
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
    )
    sequential_model.save(training_context.model_output_file_path)
    return history


def create_lstm_model(training_context: ModelTrainingContext):
    """
    :return: LSTM model with 1 output layer
    """
    lstm_model = Sequential()
    lstm_layer = LSTM(
        units=250,
        activation="tanh",
        input_shape=training_context.input_shape,
        kernel_regularizer=l2(0.001),
    )
    lstm_model.add(lstm_layer)
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["accuracy"])
    logging.info(lstm_model.summary())
    return lstm_model


def create_gru_model(training_context: ModelTrainingContext):
    """
    :return: GRU model with 1 output layer
    """
    gru_model = Sequential()
    gru_layer = GRU(
        units=250,
        activation="tanh",
        input_shape=training_context.input_shape,
        kernel_regularizer=l2(0.001),
    )
    gru_model.add(gru_layer)
    gru_model.add(Dropout(0.2))
    gru_model.add(Dense(units=1))
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["accuracy"])
    logging.info(gru_model.summary())
    return gru_model


def plot_predictions(test_dataset: np.ndarray, prediction_vector: np.ndarray):
    """
    Show the model convergence
    """
    plt.plot(test_dataset, color="gray", label="Real")
    plt.plot(prediction_vector, color="red", label="Predicted")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def plot_metrics(history: History):
    """
    Show how good the model is.
    """
    plt.plot(history.history["loss"], color="red")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(history.history["accuracy"], color="green")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def print_rmse(test_dataset: np.ndarray, predicted: np.ndarray):
    """
    Print mean squared error after training
    """
    mse = mean_squared_error(test_dataset, predicted)
    logging.info("The mean squared error is %.2f.", mse)
    rmse = np.sqrt(mse)
    logging.info("The root mean squared error is %.2f.", rmse)


def build_lstm_model(training_context: ModelTrainingContext):
    """
    Create new model, train it, evaluate results and make prediction
    """
    lstm_model = create_lstm_model(training_context)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = train_model(training_context, lstm_model, [early_stopping])

    if app_config.is_gui_enabled():
        plot_metrics(history)

    lstm_predicted_stock_price = lstm_model.predict(training_context.x_test_stock_price_dataset)

    loss, accuracy = lstm_model.evaluate(
        training_context.x_test_stock_price_dataset, training_context.y_test_stock_price_dataset
    )
    logging.info("LSTM loss=%s accuracy=%s", loss, accuracy)
    print_rmse(training_context.y_test_stock_price_dataset, lstm_predicted_stock_price)

    if app_config.is_gui_enabled():
        plot_predictions(training_context.y_test_stock_price_dataset, lstm_predicted_stock_price)


def build_gru_model(training_context: ModelTrainingContext):
    """
    Create new model, train it, evaluate results and make prediction
    """
    gru_model = create_gru_model(training_context)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = train_model(training_context, gru_model, [early_stopping])

    if app_config.is_gui_enabled():
        plot_metrics(history)

    gru_predicted_stock_price = gru_model.predict(training_context.x_test_stock_price_dataset)

    loss, accuracy = gru_model.evaluate(
        training_context.x_test_stock_price_dataset, training_context.y_test_stock_price_dataset
    )
    logging.info("GRU loss=%s accuracy=%s", loss, accuracy)
    print_rmse(training_context.y_test_stock_price_dataset, gru_predicted_stock_price)

    if app_config.is_gui_enabled():
        plot_predictions(training_context.y_test_stock_price_dataset, gru_predicted_stock_price)


def choose_model_build_strategy(
    training_context: ModelTrainingContext,
) -> Callable[[ModelTrainingContext], None]:
    """
    Build LSTM or GRU (default is GRU) model and make prediction.
    """

    match training_context.model_type:
        case "LSTM":
            model_builder_fn = build_lstm_model
        case "GRU":
            model_builder_fn = build_gru_model
        case _:
            model_builder_fn = build_gru_model
    return model_builder_fn


def reshape_with_features(dataset_to_reshape: np.ndarray, features: int):
    """
    :return: dataset with features append
    """
    rows, cols, _ = dataset_to_reshape.shape
    return dataset_to_reshape.reshape(rows, cols, features)


_YEAR_START = 2013
_YEAR_END = 2020

_STEPS = 60
_FEATURES = 1


def prepare_training_context(model_type: str):
    """
    Split dataset to train and test
    """
    np.random.seed(455)
    tf.random.set_seed(455)

    dataset = load_dataset()

    if app_config.is_gui_enabled():
        plot_train_test(dataset, _YEAR_START, _YEAR_END)

    training_set, test_set = train_test_split(dataset, _YEAR_START, _YEAR_END)

    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set = training_set.reshape(-1, 1)
    scaled_training_set = scaler.fit_transform(training_set)

    x_train_matrix, y_train_matrix = split_sequence(scaled_training_set, _STEPS)
    x_train_matrix = reshape_with_features(x_train_matrix, _FEATURES)

    dataset_total: pd.DataFrame = dataset.loc[:, "High"]
    test_dataset_fraction = len(dataset_total) - len(test_set) - _STEPS
    x_test_set: np.ndarray = dataset_total[test_dataset_fraction:].values
    x_test_set = x_test_set.reshape(-1, 1)
    scaled_test_set = scaler.transform(x_test_set)

    x_test_dataset, y_test_dataset = split_sequence(scaled_test_set, _STEPS)

    return ModelTrainingContext(
        model_type=model_type,
        scaler=scaler,
        x_train_stock_price_dataset=x_train_matrix,
        y_train_stock_price_dataset=y_train_matrix,
        input_shape=(_STEPS, _FEATURES),
        x_test_stock_price_dataset=x_test_dataset,
        y_test_stock_price_dataset=y_test_dataset,
        model_output_file_path=app_config.MODEL_OUTPUT_PATH,
    )


_NEEDS_TO_BUILD_ARG_INDEX = 1
_MODEL_TYPE_ARG_INDEX = 2


def main():
    """
    Build model or load and make a prediction.
    """
    needs_to_build = sys.argv[_NEEDS_TO_BUILD_ARG_INDEX]
    model_type_arg = sys.argv[_MODEL_TYPE_ARG_INDEX].upper()

    if needs_to_build.lower() == "true":
        context = prepare_training_context(model_type_arg)
        model_builder = choose_model_build_strategy(context)
        model_builder(context)
    else:
        prices = np.array(
            [164.36000061035156, 166.50999450683594, 166.47000122070312, 167.64999389648438]
        )
        prices = np.ndarray(shape=(1, 1), buffer=prices)

        predicted_stock_price = predict.predict(prices, model_type_arg)
        print_rmse(prices, predicted_stock_price)


if __name__ == "__main__":
    main()
