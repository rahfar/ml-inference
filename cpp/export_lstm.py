"""Export the trained LSTM to TorchScript for the C++ server.

    uv run python cpp/export_lstm.py      # models/weights/lstm.pt -> lstm.ts

TorchScript runs the same ATen kernels as Python's eager mode. Also tried:
AOTInductor (torch.export, compiled ahead of time, with oneDNN disabled so the
LSTM is lowered to matmuls). It was 10% faster at batch 1 but 3-17% slower
from batch 4 up, where batching works, than torch's fused LSTM kernel.

The export takes the relative history (n, 30, 5) and returns offsets (n, 30);
the pre/post-processing of models/data.py is done in C++.
"""

import warnings

import torch

from models.data import HISTORY_FEATURES, HISTORY_STEPS
from models.lstm import WEIGHTS, Net

TORCHSCRIPT = WEIGHTS.with_suffix(".ts")


def main() -> None:
    net = Net()
    net.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    net.eval()
    check = torch.randn(64, HISTORY_STEPS, HISTORY_FEATURES)
    with warnings.catch_warnings():  # "not supported in Python 3.14+"; works here
        warnings.simplefilter("ignore", FutureWarning)
        traced = torch.jit.trace(net, torch.randn(4, HISTORY_STEPS, HISTORY_FEATURES))
        traced.save(str(TORCHSCRIPT))
        with torch.inference_mode():
            torch.testing.assert_close(
                torch.jit.load(str(TORCHSCRIPT))(check), net(check)
            )
    print(f"wrote {TORCHSCRIPT}")


if __name__ == "__main__":
    main()
