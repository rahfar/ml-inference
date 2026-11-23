BENCH   ?= latency
SERVER  ?= all
SERVERS ?= fastapi grpc fastapi_queue fastapi_msgpack go fastapi_batched

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

# Quick run from the host against already-running servers (published ports).
# Client and servers share CPUs here -- use bench-docker for real numbers.
bench:
	uv run python runner.py --bench $(BENCH) --server $(SERVER)

build-bench:
	docker compose -f compose/bench.yml build

# Isolated run: for each server, start a fresh container, run the client on the
# compose network with its own cpuset, tear the server down. One JSON per
# server in results/, merged into results/$(BENCH).html.
bench-docker: build-bench
	@mkdir -p results
	@for s in $(SERVERS); do \
	  $(MAKE) --no-print-directory up-$$s || exit 1; \
	  docker compose -f compose/bench.yml run --rm bench \
	    --bench $(BENCH) --server $$s --output json --out results/$(BENCH)-$$s.json; \
	  status=$$?; \
	  $(MAKE) --no-print-directory down-$$s; \
	  [ $$status -eq 0 ] || exit $$status; \
	done
	uv run python report.py $(foreach s,$(SERVERS),results/$(BENCH)-$(s).json)
	uv run python report.py $(foreach s,$(SERVERS),results/$(BENCH)-$(s).json) --output html --out results/$(BENCH).html

compare:
	$(MAKE) bench-docker BENCH=concurrency

bench-all:
	$(MAKE) bench-docker BENCH=all

train:
	uv run python model/train.py

proto:
	uv run python -m grpc_tools.protoc -I servers/grpc --python_out=servers/grpc --grpc_python_out=servers/grpc servers/grpc/inference.proto
	sed -i '' 's/^import inference_pb2/from servers.grpc import inference_pb2/' servers/grpc/inference_pb2_grpc.py

.PHONY: build up down logs bench build-bench bench-docker compare bench-all train proto style
