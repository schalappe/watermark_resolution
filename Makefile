export PYTHONPATH=$(shell pwd)

VIRTUAL_ENV=venv
PYTHON=${VIRTUAL_ENV}/bin/python
PIP=${VIRTUAL_ENV}/bin/pip

.PHONY: get_data
.ONESHELL: get_data

create_virtualenv: requirements.txt
	python3 -m venv $(VIRTUAL_ENV)
	$(PIP) install -r requirements.txt

get_data:
	rm -rf .env
	mkdir -p data/images
	@echo DATA_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_aesthetic_10k&config=default&split=train" >> .env
	@echo RAW_PATH=$(shell pwd)/data/ >> .env
	$(PYTHON) src/data/build_dataset.py

