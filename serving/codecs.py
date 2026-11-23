"""Wire formats: every codec turns a float32 array into bytes and back.

The server and the bench client use the same functions. decode() may return
any shape; the server reshapes requests to (-1, 150) and the client doesn't
care about the response shape.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import msgpack
import numpy as np

from serving import tensor_pb2


@dataclass(frozen=True)
class Codec:
    content_type: str
    encode: Callable[[np.ndarray], bytes]
    decode: Callable[[bytes], np.ndarray]


# Text: a nested list of numbers, the way most JSON APIs look.
def json_encode(a: np.ndarray) -> bytes:
    return json.dumps(a.tolist()).encode()


def json_decode(b: bytes) -> np.ndarray:
    return np.asarray(json.loads(b), dtype=np.float32)


# Binary, but still a nested list of numbers (each float packed separately).
def msgpack_encode(a: np.ndarray) -> bytes:
    return msgpack.packb(a.tolist())


def msgpack_decode(b: bytes) -> np.ndarray:
    return np.asarray(msgpack.unpackb(b), dtype=np.float32)


# What gRPC sends: a protobuf message with a packed repeated float field.
def protobuf_encode(a: np.ndarray) -> bytes:
    return tensor_pb2.Tensor(values=a.ravel().tolist()).SerializeToString()


def protobuf_decode(b: bytes) -> np.ndarray:
    message = tensor_pb2.Tensor()
    message.ParseFromString(b)
    return np.asarray(message.values, dtype=np.float32)


# The lower bound: the array's memory as-is, nothing to parse.
def raw_encode(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float32).tobytes()


def raw_decode(b: bytes) -> np.ndarray:
    # copy: frombuffer gives a read-only view and torch wants a writable array
    return np.frombuffer(b, dtype=np.float32).copy()


CODECS = {
    "json": Codec("application/json", json_encode, json_decode),
    "msgpack": Codec("application/msgpack", msgpack_encode, msgpack_decode),
    "protobuf": Codec("application/x-protobuf", protobuf_encode, protobuf_decode),
    "raw": Codec("application/octet-stream", raw_encode, raw_decode),
}
