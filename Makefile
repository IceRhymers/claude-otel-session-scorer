.PHONY: setup clean build test install

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
	$(POETRY) run pytest tests/

install: setup
	$(POETRY) install

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies using Poetry"
	@echo "  make build      - Build the package using Poetry"
	@echo "  make clean      - Remove build artifacts and Poetry environment"
	@echo "  make test       - Run tests"
	@echo "  make install    - Install package in development mode"
	@echo "  make help       - Show this help message"
