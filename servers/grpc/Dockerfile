FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy grpcio grpcio-tools

COPY model/ model/
COPY servers/__init__.py servers/__init__.py
COPY servers/grpc/ servers/grpc/

ENV PYTHONPATH=/app

EXPOSE 50051

CMD ["python", "servers/grpc/server.py", "--host", "0.0.0.0", "--port", "50051"]
