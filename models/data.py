"""Synthetic vessel tracks, shared by training, profiling and the bench client."""

import numpy as np

HISTORY_STEPS = 30  # 5 min at 10-s intervals
HISTORY_FEATURES = 5  # lat, lon, speed, sin(course), cos(course)
FUTURE_STEPS = 15  # 15 min at 1-min intervals
FUTURE_FEATURES = 2  # lat, lon

INPUT_SIZE = HISTORY_STEPS * HISTORY_FEATURES  # 150 floats per sample
OUTPUT_SIZE = FUTURE_STEPS * FUTURE_FEATURES  # 30 floats per sample


def make_dataset(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Near-straight tracks with noisy speed and course.

    Returns X: (n, 150) and y: (n, 30), float32, flattened row-major.
    """
    rng = np.random.default_rng(seed)
    lat = rng.uniform(54.0, 64.0, n)
    lon = rng.uniform(0.0, 20.0, n)
    speed = rng.uniform(5.0, 20.0, n)
    course = rng.uniform(0.0, 2 * np.pi, n)

    def advance(hours: float) -> None:
        nonlocal lat, lon
        lat = lat + speed * hours * np.cos(course)
        lon = lon + speed * hours * np.sin(course) / np.maximum(
            np.cos(np.radians(lat)), 0.01
        )

    history = np.empty((n, HISTORY_STEPS, HISTORY_FEATURES), np.float32)
    for t in range(HISTORY_STEPS):
        history[:, t] = np.stack(
            [lat, lon, speed, np.sin(course), np.cos(course)], axis=1
        )
        advance(10 / 3600)
        speed = speed + rng.normal(0, 0.05, n)
        course = course + rng.normal(0, 0.005, n)

    future = np.empty((n, FUTURE_STEPS, FUTURE_FEATURES), np.float32)
    for t in range(FUTURE_STEPS):
        advance(60 / 3600)
        speed = speed + rng.normal(0, 0.1, n)
        course = course + rng.normal(0, 0.02, n)
        future[:, t] = np.stack([lat, lon], axis=1)

    return history.reshape(n, INPUT_SIZE), future.reshape(n, OUTPUT_SIZE)
