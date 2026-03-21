FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer-caching
COPY pyproject.toml uv.lock ./

# Use Python 3.12 (override .python-version which may specify a newer version)
# UV_PYTHON_DOWNLOADS=never prevents uv from fetching a different interpreter
RUN echo "3.12" > .python-version && \
    UV_PYTHON_DOWNLOADS=never uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Copy source code (model weights are mounted at runtime)
COPY *.py ./

EXPOSE 8000 8001
