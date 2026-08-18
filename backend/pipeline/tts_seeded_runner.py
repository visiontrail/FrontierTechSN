"""Run a vendor TTS script with reproducible random-number state.

VibeVoice's language-model decode is greedy, but its diffusion decoder still
consults PyTorch's RNG. Keeping this wrapper in the application repository lets
every chunk start from the same seed without modifying the installed vendor
checkout under ``AIWORK_ROOT``.
"""

from __future__ import annotations

import random
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: tts_seeded_runner.py SEED SCRIPT [SCRIPT_ARGS...]")

    seed = int(sys.argv[1])
    script = Path(sys.argv[2]).expanduser().resolve()
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
