FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy fastapi uvicorn[standard]

COPY model/ model/
COPY servers/fastapi/app.py servers/fastapi/app.py

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "servers/fastapi/app.py", "--host", "0.0.0.0", "--port", "8000"]
