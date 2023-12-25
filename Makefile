export PYTHONPATH=$(shell pwd)

VIRTUAL_ENV=$(shell pwd)/.venv
PYTHON=${VIRTUAL_ENV}/bin/python
JUPYTER=${VIRTUAL_ENV}/bin/jupyter-lab
OPTUNA=${VIRTUAL_ENV}/bin/optuna-dashboard
PIP=${VIRTUAL_ENV}/bin/pip

.PHONY: train_model search_loss search_model initialize get_data notebook
.ONESHELL: initialize get_data

initialize:
	rm -rf .env
	mkdir -p data/train
	mkdir -p data/tests
	mkdir -p models/storage
	mkdir -p models/params
	@echo RAW_PATH=$(shell pwd)/data >> .env
	@echo MODELS_PATH=$(shell pwd)/models >> .env

create_virtualenv: requirements.txt
	python3 -m venv $(VIRTUAL_ENV)
	$(PIP) install -r requirements.txt

notebook:
	cd notebooks/ && $(JUPYTER) --port=8080

dashboard:
	$(OPTUNA) sqlite:///$(shell pwd)/models/params/watermark.db

get_data:
	@echo TRAIN_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_aesthetic_10k&config=default&split=train" >> .env
	@echo TESTS_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_6k&config=default&split=train" >> .env
	$(PYTHON) src/data/build_dataset.py

search_model:
	$(PYTHON) src/scripts/search_parameters_model.py

search_loss:
	$(PYTHON) src/scripts/search_parameters_loss.py

train_model:
	$(PYTHON) src/scripts/train_model.py
