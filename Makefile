# Makefile for AI Trading System

# Command to run linting
lint:
	flake8 .

# Command to run tests
test:
	pytest

# Command to start development server
serve:
	flask run

# All commands
all: lint test serve
