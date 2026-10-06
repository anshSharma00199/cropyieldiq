.PHONY: install train test lint security run up down scale
install: ; pip install -r requirements-dev.txt
train:   ; python ml/train.py
test:    train ; pytest --cov=app --cov-report=term-missing
lint:    ; ruff check .
security: ; bandit -r app -ll && pip-audit -r requirements.txt
run:     train ; uvicorn app.main:app --reload --port 8000
up:      ; docker compose up -d --build
down:    ; docker compose down
scale:   ; docker compose up -d --scale api=4
