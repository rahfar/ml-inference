EXP   ?= strategy
MODEL ?= lstm

train:            ## train the models -> models/weights/
	uv run python -m models.train

profile:          ## model-only cost vs batch size
	uv run python -m models.profile --model $(MODEL)

serve:            ## run one server by hand, e.g. make serve ARGS="--strategy serial"
	uv run python -m serving.app $(ARGS)

bench:            ## run one experiment from experiments.yaml, e.g. make bench EXP=codec
	uv run python -m bench.run $(EXP)

proto:            ## regenerate serving/tensor_pb2.py
	uv run python -m grpc_tools.protoc -I serving --python_out=serving --pyi_out=serving serving/tensor.proto

style:            ## fix what can be fixed: imports, lint autofixes, formatting
	uv run ruff check --fix .
	uv run ruff format .

lint:             ## check only, changes nothing: ruff lint + format, ty types
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

.PHONY: train profile serve bench proto style lint
