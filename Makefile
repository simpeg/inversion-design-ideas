PYTEST_TARGETS=src tests
PYTEST_ARGS=--cov-report=term-missing --cov --doctest-modules --verbose
CHECK_STYLE=src notebooks tests
CHECK_TYPE=src

.PHONY: help install test check check-format check_style format mypy

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  check     run code style and quality checks with Ruff"
	@echo "  format    automatically format the code with Ruff"
	@echo "  mypy      run type checks with mypy"
	@echo ""

install:
	python -m pip install --no-deps --editable .

test:
	pytest $(PYTEST_ARGS) $(PYTEST_TARGETS)

check: check-format check-style

check-format:
	ruff format --check $(CHECK_STYLE)

check-style:
	ruff check $(CHECK_STYLE)

mypy:
	mypy $(CHECK_TYPE)

format:
	ruff format $(CHECK_STYLE)
	ruff check --fix $(CHECK_STYLE)
