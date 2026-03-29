"""Build fixed test payloads for a given batch size."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model_def import HISTORY_STEPS


def _base_track(n: int = HISTORY_STEPS) -> list[dict]:
    return [
        {
            "lat": 58.0 + i * 0.001,
            "lon": 10.0,
            "speed": 12.0,
            "course_sin": 0.5,
            "course_cos": 0.866,
        }
        for i in range(n)
    ]


def http_payload(batch_size: int) -> dict:
    track = _base_track()
    return {"vessels": [{"history": track}] * batch_size}


def grpc_request(batch_size: int):
    import inference_pb2

    track = _base_track()
    single = inference_pb2.PredictRequest(
        history=[
            inference_pb2.TrackPoint(
                lat=p["lat"],
                lon=p["lon"],
                speed=p["speed"],
                course_sin=p["course_sin"],
                course_cos=p["course_cos"],
            )
            for p in track
        ]
    )
    return inference_pb2.PredictBatchRequest(vessels=[single] * batch_size)
