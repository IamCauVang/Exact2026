from __future__ import annotations

# Thin wrapper so you can run benchmark locally without remembering the core runner flags.
# Example:
#   python scripts/benchmark.py --dataset data/fraction_dataset.json --input-mode nl --batch-size 16 --limit 100

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_core_runner import main

if __name__ == "__main__":
    main()
