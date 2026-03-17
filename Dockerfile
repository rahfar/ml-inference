FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (avoids pulling the massive CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" \
    flask waitress \
    psutil httpx \
    grpcio grpcio-tools \
    numpy catboost

# Copy source (model artifacts excluded via .dockerignore)
COPY . .

# Bake trained model into the image so containers start instantly
RUN python train_pytorch.py

EXPOSE 8000 50051

# Override CMD at runtime: python server_fastapi.py / server_flask.py / server_grpc.py
CMD ["python", "server_fastapi.py"]
