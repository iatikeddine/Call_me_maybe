.PHONY: all run clean fclean re

UV      = uv
VENV    = .venv

all: $(VENV)

$(VENV):
	$(UV) sync

run: $(VENV)
	$(UV) run python -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

fclean: clean
	rm -rf $(VENV)
	rm -f data/output/function_calling_results.json

re: fclean all
