BENCH   ?= latency
SERVER  ?= all

COMPOSE_FILES := $(wildcard compose/compose.*.yml)
COMPOSE_FLAGS := $(foreach f,$(COMPOSE_FILES),-f $(f))

style:
	uv run ruff check --fix --select I .
	uv run ruff format .


build:
	docker compose $(COMPOSE_FLAGS) build

build-%:
	docker compose -f compose/compose.$*.yml build

up:
	docker compose $(COMPOSE_FLAGS) up -d --wait

up-%:
	docker compose -f compose/compose.$*.yml up -d --wait

down:
	docker compose $(COMPOSE_FLAGS) down

down-%:
	docker compose -f compose/compose.$*.yml down

logs:
	docker compose $(COMPOSE_FLAGS) logs -f

logs-%:
	docker compose -f compose/compose.$*.yml logs -f

bench:
	python runner.py --bench $(BENCH) --server $(SERVER)

compare: up
	python runner.py --bench concurrency --server all --output html

bench-all: up
	python runner.py --bench all --server all --output html

train:
	cd model && python train.py

proto:
	uv run python -m grpc_tools.protoc -I servers/grpc --python_out=servers/grpc --grpc_python_out=servers/grpc servers/grpc/inference.proto

.PHONY: build up down logs bench compare bench-all train proto style
