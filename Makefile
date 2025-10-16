PYTEST_TARGETS=src
CHECK_STYLE=src notebooks

.PHONY: help install test check check-format check_style format

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  check     run code style and quality checks with Ruff"
	@echo "  format    automatically format the code with Ruff"
	@echo ""

install:
	python -m pip install --no-deps --editable .

test:
	pytest --verbose --doctest-modules $(PYTEST_TARGETS)

check: check-format check-style

check-format:
	ruff format --check $(CHECK_STYLE)

check-style:
	ruff check $(CHECK_STYLE)

format:
	ruff check --fix $(CHECK_STYLE)
	ruff format $(CHECK_STYLE)

