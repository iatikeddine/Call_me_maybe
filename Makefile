.PHONY: install run debug clean lint

DEFAULT = --functions_definition data/input/functions_definition.json \
			--input data/input/function_calling_tests.json \
			--output data/output/function_calling_results
VENV    = .venv

install: $(VENV)

$(VENV): pyproject.toml uv.lock
	uv sync

run: $(VENV)
	uv run python -m src $(DEFAULT)

debug: $(VENV)
	uv run python -m pdb -m src $(DEFAULT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint: $(VENV)
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
