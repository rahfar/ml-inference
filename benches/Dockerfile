# Load-generator image: bench code only, no torch.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir aiohttp uvloop msgpack grpcio protobuf pyyaml

COPY benches/ benches/
COPY servers/__init__.py servers/__init__.py
COPY servers/grpc/__init__.py servers/grpc/inference_pb2.py servers/grpc/inference_pb2_grpc.py servers/grpc/
COPY runner.py report.py config.yaml ./

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "runner.py", "--target", "docker"]
