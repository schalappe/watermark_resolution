export PYTHONPATH=$(shell pwd)

MODELS_DB=sqlite:///$(shell pwd)/models/params/watermark.db

.PHONY: train_model search_loss search_model initialize get_data notebook dashboard sync
.ONESHELL: initialize get_data

initialize:
	rm -rf .env
	mkdir -p data/train
	mkdir -p data/tests
	mkdir -p models/storage
	mkdir -p models/params
	@echo RAW_PATH=$(shell pwd)/data >> .env
	@echo MODELS_PATH=$(shell pwd)/models >> .env

sync:
	uv sync --dev

notebook:
	cd notebooks/ && uv run jupyter-lab --port=8080

dashboard:
	uv run optuna-dashboard $(MODELS_DB)

get_data:
	@echo TRAIN_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_aesthetic_10k&config=default&split=train" >> .env
	@echo TESTS_URL="https://datasets-server.huggingface.co/rows?dataset=ioclab%2Fgrayscale_image_6k&config=default&split=train" >> .env
	uv run python src/data/build_dataset.py

search_model:
	uv run python src/scripts/search_parameters_model.py

search_loss:
	uv run python src/scripts/search_parameters_loss.py

train_model:
	uv run python src/scripts/train_model.py
