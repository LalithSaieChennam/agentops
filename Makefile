.PHONY: install dev lint test train seed simulate demo up down clean

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Run linter
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Seed training data
seed:
	python scripts/seed_data.py

# Train initial model
train:
	python scripts/initial_train.py

# Simulate drift
simulate:
	python scripts/simulate_drift.py

# Run full demo
demo:
	python scripts/demo.py

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Run the API locally (without Docker)
run:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Run MCP server
mcp:
	python -m src.mcp.server

# Clean artifacts
clean:
	rm -rf models/best models/backup_* data/processed/* data/raw/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
