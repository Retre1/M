.PHONY: install train backtest live dashboard lint test clean

install:
	pip install -e ".[all]"

train:
	python scripts/train.py $(ARGS)

backtest:
	python scripts/backtest.py $(ARGS)

live:
	python scripts/live_trade.py $(ARGS)

dashboard:
	python scripts/launch_dashboard.py

collect:
	python scripts/collect_data.py $(ARGS)

lint:
	ruff check src/ tests/
	mypy src/apexfx/

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/apexfx --cov-report=html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
