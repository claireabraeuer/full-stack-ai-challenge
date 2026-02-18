.PHONY: help install dev-install clean test lint format run-api train docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies with uv"
	@echo "  make dev-install   - Install dev dependencies"
	@echo "  make clean         - Clean up generated files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code with black and ruff"
	@echo "  make run-api       - Run the FastAPI server"
	@echo "  make train         - Train models"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"

install:
	uv pip install -e .

dev-install:
	uv pip install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov dist build

test:
	pytest

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

train:
	python -m src.models.train

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down
