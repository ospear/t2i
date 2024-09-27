include .env

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: fmt
fmt:
	uv run ruff format .

.PHONY: vet
vet:
	uv run pyright .

.PHONY: dev
dev:
	uv run uvicorn t2i.app:app --reload

.PHONY: run
run:
	uv run uvicorn t2i.app:app --workers=2

.PHONY: preload_models
preload_models:
	uv run python main.py preload_models

.PHONY: build
build:
	docker build --target production --tag ospear/t2i:latest --build-arg HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) ./

.PHONY: push
push:
	docker push ospear/t2i:latest
