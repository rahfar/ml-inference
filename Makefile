BENCH   ?= latency
SERVER  ?= all

style:
	uv run ruff check --fix --select I .
	uv run ruff format .
	

build:
	docker compose -f compose/docker-compose.yml build

up:
	docker compose -f compose/docker-compose.yml up -d --wait

down:
	docker compose -f compose/docker-compose.yml down

logs:
	docker compose -f compose/docker-compose.yml logs -f

bench:
	python runner.py --bench $(BENCH) --server $(SERVER)

compare: up
	python runner.py --bench concurrency --server all --output html

bench-all: up
	python runner.py --bench all --server all --output html

train:
	cd model && python train.py

proto:
	python -m grpc_tools.protoc -I servers/grpc --python_out=servers/grpc --grpc_python_out=servers/grpc servers/grpc/inference.proto

.PHONY: build up down logs bench compare bench-all train proto style
