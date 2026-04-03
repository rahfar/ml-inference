# ── Build stage ────────────────────────────────────────────────────────────────
FROM golang:1.22-bookworm AS builder

WORKDIR /app

# Copy module file first; go mod tidy will fetch deps and write go.sum.
COPY servers/go/go.mod .
COPY servers/go/main.go .

# GONOSUMDB=* skips the checksum DB so go mod tidy works in air-gapped builds.
# CGO_ENABLED=1 is required by onnxruntime_go (uses cgo for dlopen).
RUN GONOSUMDB=* GOFLAGS=-mod=mod go mod tidy && \
    CGO_ENABLED=1 go build -o server .

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

ARG ORT_VERSION=1.19.2

# Download the ONNX Runtime prebuilt shared library for Linux x86-64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget \
    && wget -qO /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /app/lib \
    && tar xzf /tmp/ort.tgz -C /tmp \
    && cp /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* /app/lib/ \
    && rm -rf /tmp/ort.tgz "/tmp/onnxruntime-linux-x64-${ORT_VERSION}" \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/server /app/server

EXPOSE 8000

CMD ["/app/server", \
     "-model", "/app/weights/model.onnx", \
     "-lib",   "/app/lib/libonnxruntime.so"]
