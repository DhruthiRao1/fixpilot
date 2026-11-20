.PHONY: install test eval run adk lint docker

install:
	pip install -e ".[dev,adk]"

test:
	pytest -q

eval:
	python evals/run_eval.py

run:
	uvicorn fixpilot.api.main:app --reload --port 8080

adk:
	cd src/fixpilot && adk web

lint:
	ruff check src tests evals

docker:
	docker build -t fixpilot:local . && docker run --rm -p 8080:8080 fixpilot:local
