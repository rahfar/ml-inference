"""The server: one handler (serving/handler.py) behind a choice of web stacks.

    uv run python -m serving.app --model lstm --strategy batched --codec json
    uv run python -m serving.app --stack flask+gunicorn --model catboost

POST /predict   body: (n, 150) float32 in the chosen codec -> (n, 30) float32
GET  /health

A stack is a framework (FastAPI, Starlette: async/ASGI; Flask: sync/WSGI) run
by an HTTP server (uvicorn: ASGI; gunicorn: WSGI; granian: both). The CLI puts
the settings in environment variables and replaces itself with the server.

WSGI apps handle each request in one of the server's threads, which is exactly
the inline_threadpool strategy, so that's the only strategy they support.
"""

import argparse
import os
import sys

from fastapi import FastAPI, Request
from fastapi import Response as FastAPIResponse
from flask import Flask
from flask import Response as FlaskResponse
from flask import request as flask_request
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import models
from serving.codecs import CODECS
from serving.handler import Handler
from serving.strategies import STRATEGIES

# Threads for request handling, everywhere: Starlette's default thread pool
# (used by inline_threadpool) has 40, so the WSGI servers get 40 too.
REQUEST_THREADS = 40


def fastapi_app() -> FastAPI:
    handler = Handler.from_env()
    app = FastAPI(lifespan=handler.lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(request: Request) -> FastAPIResponse:
        reply = await handler.handle(await request.body())
        return FastAPIResponse(
            reply.body, reply.status, reply.headers, reply.content_type
        )

    return app


def starlette_app() -> Starlette:
    handler = Handler.from_env()

    async def health(_: StarletteRequest) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def predict(request: StarletteRequest) -> Response:
        reply = await handler.handle(await request.body())
        return Response(reply.body, reply.status, reply.headers, reply.content_type)

    routes = [Route("/health", health), Route("/predict", predict, methods=["POST"])]
    return Starlette(routes=routes, lifespan=handler.lifespan)


def flask_app() -> Flask:
    handler = Handler.from_env()
    app = Flask(__name__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    def predict() -> FlaskResponse:
        reply = handler.handle_sync(flask_request.get_data())
        return FlaskResponse(
            reply.body, reply.status, reply.headers, content_type=reply.content_type
        )

    return app


STACKS = {  # stack -> (app factory, server)
    "fastapi+uvicorn": ("fastapi_app", "uvicorn"),
    "fastapi+granian": ("fastapi_app", "granian"),
    "starlette+uvicorn": ("starlette_app", "uvicorn"),
    "starlette+granian": ("starlette_app", "granian"),
    "flask+gunicorn": ("flask_app", "gunicorn"),
    "flask+granian": ("flask_app", "granian"),
}
WSGI_APPS = {"flask_app"}


def server_command(factory: str, server: str, host: str, port: int) -> list[str]:
    """The command line that runs `factory` on `server`, one process."""
    target = f"serving.app:{factory}"
    wsgi = factory in WSGI_APPS
    if server == "uvicorn":
        return [
            "uvicorn",
            target,
            "--factory",
            f"--host={host}",
            f"--port={port}",
            "--log-level=warning",
        ]
    if server == "granian":
        cmd = [
            "granian",
            target,
            "--factory",
            f"--interface={'wsgi' if wsgi else 'asgi'}",
            f"--host={host}",
            f"--port={port}",
            "--workers=1",
            "--log-level=warning",
        ]
        return cmd + ([f"--blocking-threads={REQUEST_THREADS}"] if wsgi else [])
    if server == "gunicorn":
        return [
            "gunicorn",
            f"{target}()",
            "--worker-class=gthread",
            "--workers=1",
            f"--threads={REQUEST_THREADS}",
            f"--bind={host}:{port}",
            "--graceful-timeout=5",
            "--log-level=warning",
        ]
    raise ValueError(f"unknown server {server!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inference server")
    p.add_argument("--stack", choices=STACKS, default="fastapi+uvicorn")
    p.add_argument("--model", choices=models.MODELS, default="lstm")
    p.add_argument("--strategy", choices=STRATEGIES, default="batched")
    p.add_argument("--codec", choices=CODECS, default="json")
    p.add_argument("--threads", type=int, default=4, help="threads per predict() call")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    factory, server = STACKS[args.stack]
    if factory in WSGI_APPS and args.strategy != "inline_threadpool":
        p.error(f"{args.stack} is WSGI: only --strategy inline_threadpool applies")

    os.environ.update(
        SERVING_MODEL=args.model,
        SERVING_STRATEGY=args.strategy,
        SERVING_CODEC=args.codec,
        SERVING_THREADS=str(args.threads),
    )
    cmd = [sys.executable, "-m", *server_command(factory, server, args.host, args.port)]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
