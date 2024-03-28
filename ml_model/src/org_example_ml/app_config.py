"""
Holds the configuration for model training.
"""

import os


GUI_ENABLED = os.getenv("APP_GUI_ENABLED")

DATASET_FOLDER_PATH = os.getenv("APP_DATASET_FOLDER_PATH")
DATASET_PATH = os.path.join(f"{DATASET_FOLDER_PATH}/stock_history.csv")

LSTM_MODEL_PATH = os.getenv("APP_LSTM_MODEL_PATH")
GRU_MODEL_PATH = os.getenv("APP_GRU_MODEL_PATH")
MODEL_OUTPUT_PATH = os.getenv("APP_MODEL_OUTPUT_PATH", "model.keras")



def is_gui_enabled():
    return GUI_ENABLED.capitalize() == "True"
