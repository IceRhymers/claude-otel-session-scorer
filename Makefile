.PHONY: setup clean build test install lint check fmt

POETRY = poetry

setup:
	$(POETRY) install

build: setup
	$(POETRY) build

clean:
	rm -rf dist/
	rm -rf *.egg-info/
	$(POETRY) env remove --all

test: setup
	$(POETRY) run pytest tests/ -v

install: setup
	$(POETRY) install

lint:
	ruff check . && ruff format --check .

check: lint

fmt:
	ruff format .

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies using Poetry"
	@echo "  make build      - Build the package using Poetry"
	@echo "  make clean      - Remove build artifacts and Poetry environment"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run ruff lint and format checks"
	@echo "  make check      - Alias for lint (CI gate)"
	@echo "  make fmt        - Auto-fix formatting with ruff"
	@echo "  make install    - Install package in development mode"
	@echo "  make help       - Show this help message"
