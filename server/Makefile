setup:
	python -m pip install --upgrade pip
	pip install pipenv

i:
	pipenv install
run:
	export APP_LSTM_MODEL_PATH="$(PWD)/models/lstm_model.keras" && \
	export APP_GRU_MODEL_PATH="$(PWD)/models/gru_model.keras" && \
	cd src && \
	export FLASK_APP=org_example_fin_predictor.app && \
	pipenv run python -m flask run --debug
