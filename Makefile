BENCH   ?= latency
SERVER  ?= all

style:
	uv run ruff check --fix --select I .
	uv run ruff format .

# Run one server in the foreground, e.g. `make serve-fastapi_batched`
serve-%:
	uv run python -c "import yaml, os, sys; c = yaml.safe_load(open('config.yaml'))['servers']['$*']['cmd']; os.execv(sys.executable, [sys.executable, *c])"

# Fresh server process per bench run, report -> results/<bench>.html
bench:
	@mkdir -p results
	uv run python runner.py --bench $(BENCH) --server $(SERVER) --spawn --output html --out results/$(BENCH).html

compare:
	@mkdir -p results
	uv run python runner.py --bench concurrency --server all --spawn --output html --out results/concurrency.html

bench-all:
	@mkdir -p results
	uv run python runner.py --bench all --server all --spawn --output html --out results/all.html

train:
	uv run python model/train.py

proto:
	uv run python -m grpc_tools.protoc -I servers/grpc --python_out=servers/grpc --grpc_python_out=servers/grpc servers/grpc/inference.proto
	sed -i '' 's/^import inference_pb2/from servers.grpc import inference_pb2/' servers/grpc/inference_pb2_grpc.py

.PHONY: bench compare bench-all train proto style
