export PYTHONPATH=$(shell pwd)

VIRTUAL_ENV=$(shell pwd)/.venv
PYTHON=${VIRTUAL_ENV}/bin/python
JUPYTER=${VIRTUAL_ENV}/bin/jupyter-lab
PIP=${VIRTUAL_ENV}/bin/pip

.PHONY: get_data notebook
.ONESHELL: get_data

create_virtualenv: requirements.txt
	python3 -m venv $(VIRTUAL_ENV)
	$(PIP) install -r requirements.txt

notebook:
	cd notebooks/ && $(JUPYTER) --port=8080

get_data:
	rm -rf .env
	mkdir -p data/images
	mkdir -p data/params
	@echo DATA_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_aesthetic_10k&config=default&split=train" >> .env
	@echo RAW_PATH=$(shell pwd)/data/ >> .env
	@echo RAW_PATH=$(shell pwd)/data/params >> .env
	$(PYTHON) src/data/build_dataset.py

