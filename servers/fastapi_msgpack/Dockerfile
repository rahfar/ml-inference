FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy fastapi uvicorn[standard] msgpack

COPY model/model.py model/model.py
ENV PYTHONPATH=/app/model
COPY servers/fastapi_msgpack/app.py .

EXPOSE 8000

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
