.PHONY: test

install:
	pipenv install

freeze-deps:
	pipenv pip freeze > requirements.txt

test:
	python -m unittest -v --locals

pylint:
	pylint ./src

pylint-generate-rcfile:
	pylint --generate-rcfile > .pylintrc

flake8:
	flake8 .
