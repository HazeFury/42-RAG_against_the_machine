NAME = rag-against-the-machine
VENV = .venv

all: install

install:
	@uv sync

run:
	@uv run python -m src --help

debug:
	@uv run python -m pdb -m src

clean:
	@echo "Removing temporary files and caches..."
	@rm -rf .mypy_cache .pytest_cache .ruff_cache src/*.egg-info dist build
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

fclean: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)

re: fclean all

lint:
	@echo "Running linter..."
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running strict linter..."
	@uv run flake8 src
	@uv run mypy src --strict

test:
	@echo "Running test suite..."
	@uv run pytest $(ARGS) -v

.PHONY: all install run debug clean fclean re lint lint-strict test
