.PHONY: sync data train evaluate test check

sync:
	uv sync

data:
	uv run python -m watermark.data

train:
	uv run python -m watermark.train $(ARGS)

evaluate:
	uv run python -m watermark.evaluate $(ARGS)

test:
	uv run pytest -q

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyrefly check
