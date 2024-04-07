FROM tensorflow/tensorflow:2.15.0

RUN apt update && apt upgrade -y

WORKDIR /opt/app/

COPY Pipfile .
COPY Pipfile.lock .

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv sync

COPY src ./src/

COPY models ./models/
ENV APP_LSTM_MODEL_PATH=/opt/app/models/lstm_model.keras
ENV APP_GRU_MODEL_PATH=/opt/app/models/gru_model.keras

WORKDIR /opt/app/src

EXPOSE 5000

ENV FLASK_APP=org_example_fin_predictor.app

CMD pipenv run python -m gunicorn -w 4 -b 0.0.0.0:5000 org_example_fin_predictor.app:app
