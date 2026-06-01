"""Lightweight CSV metrics logger — no extra dependencies."""

import csv
import os
import time


class MetricsLogger:
    """Writes one row per call to a CSV file and optionally prints to stdout.

    Usage
    -----
        log = MetricsLogger("logs/bc/metrics.csv")
        log.write(epoch=1, loss=0.05)
        log.close()
    """

    def __init__(self, path: str, print_keys: list[str] = None, verbose: bool = True):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        self._print_keys = print_keys
        self._verbose = verbose
        self._file = None
        self._writer = None
        self._start = time.time()

    def write(self, **kwargs) -> None:
        kwargs.setdefault("wall_time", round(time.time() - self._start, 2))
        if self._writer is None:
            self._file = open(self._path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(kwargs.keys()))
            self._writer.writeheader()
        self._writer.writerow(kwargs)
        self._file.flush()
        if self._verbose:
            keys = self._print_keys or list(kwargs.keys())
            parts = [f"{k}={kwargs[k]}" for k in keys if k in kwargs]
            print("  " + "  ".join(parts))

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
