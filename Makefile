.PHONY: doctor test test-gpu lint fmt run

doctor:   ## environment checks; must pass before any training command
	uv run slm doctor

test:     ## CPU unit tests, must finish in under 60s
	uv run pytest -m "not gpu" -q

test-gpu: ## GPU smoke tests
	uv run pytest -m gpu -q

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy src/slmkit

fmt:
	uv run ruff format . && uv run ruff check --fix .

run:      ## make run CFG=abc_music/micro_v1
	uv run slm run $(CFG)
