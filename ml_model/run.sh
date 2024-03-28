#!/bin/bash

set -e

export APP_GUI_ENABLED=false
export APP_DATASET_FOLDER_PATH="$(pwd)/data"
output_dir="$(pwd)/../server/models"
mkdir -p $output_dir
export APP_MODEL_OUTPUT_PATH="$output_dir/gru_model.keras"
export APP_GRU_MODEL_PATH=$APP_MODEL_OUTPUT_PATH

build_gru() {
  cd src
  pipenv run python -m org_example_ml.model true GRU
}

gru_predict() {
  cd src
  pipenv run python -m org_example_ml.model false GRU
}

prepare_lstm_variables() {
  export APP_MODEL_OUTPUT_PATH="$output_dir/lstm_model.keras"
  export APP_LSTM_MODEL_PATH=$APP_MODEL_OUTPUT_PATH
}

build_lstm() {
  prepare_lstm_variables
  cd src
  pipenv run python -m org_example_ml.model true LSTM
}

lstm_predict() {
  prepare_lstm_variables
  cd src
  pipenv run python -m org_example_ml.model false LSTM
}

"$@"
