setup:
	python -m pip install --upgrade pip
	pip install pipenv

i:
	pipenv install
run-dev:
	export APP_LSTM_MODEL_PATH="$(PWD)/models/lstm_model.keras" && \
	export APP_GRU_MODEL_PATH="$(PWD)/models/gru_model.keras" && \
	cd src && \
	export FLASK_APP=org_example_fin_predictor.app && \
	pipenv run python -m flask run --debug

run:
	export APP_LSTM_MODEL_PATH="$(PWD)/models/lstm_model.keras" && \
	export APP_GRU_MODEL_PATH="$(PWD)/models/gru_model.keras" && \
	cd src && \
	pipenv run python -m gunicorn \
		--bind 0.0.0.0:5000 \
		--workers 2 \
		--timeout 120 \
		--log-level=debug \
		org_example_fin_predictor.app:app
