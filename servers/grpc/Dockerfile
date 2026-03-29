FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy grpcio grpcio-tools

COPY model/model.py model/model.py
ENV PYTHONPATH=/app/model
COPY servers/grpc/inference.proto .
COPY servers/grpc/inference_pb2.py .
COPY servers/grpc/inference_pb2_grpc.py .
COPY servers/grpc/server.py .

EXPOSE 50051

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "50051"]
