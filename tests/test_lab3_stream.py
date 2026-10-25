"""ЛР3, негативные контроли: память стриминга, отказы схемы, фиксация зерна.

Тест 1: пик памяти потокового загрузчика не растёт с удвоением объёма данных.
Тест 2: pydantic-схема отвергает плохие значения (lr, dropout, архитектура, путь).
Тест 3: одинаковое зерно даёт идентичную инициализацию весов MLP.
"""
from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dl_pipeline as dp


def test_stream_memory_bounded(tmp_path: Path) -> None:
    """Пик памяти потокового загрузчика не растёт с удвоением объёма данных."""
    feats = [f"feature_{i:02d}" for i in range(20)]
    peaks = []
    for n in (4000, 8000):
        x, y = make_classification(
            n_samples=n, n_features=20, n_informative=10, n_redundant=5,
            n_repeated=5, random_state=511,
        )
        df = pd.DataFrame(x, columns=feats)
        df["target"] = y
        path = tmp_path / f"data_{n}.csv"
        df.to_csv(path, index=False)
        scaler = StandardScaler().fit(x)
        ds = dp.CsvStreamDataset(str(path), feats, np.ones(n, dtype=bool),
                                 scaler, 500, 128)
        loader = torch.utils.data.DataLoader(ds, batch_size=None)
        tracemalloc.start()
        for _ in loader:
            pass
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / 1e6)
    chunk_mb = 500 * 21 * 8 / 1e6
    assert peaks[1] - peaks[0] < 2 * chunk_mb


def test_schema_rejects_bad_config(tmp_path: Path) -> None:
    """Схема отвергает lr <= 0, dropout >= 1, пустую архитектуру, missing путь."""
    data = tmp_path / "d.csv"
    data.write_text("target\n0\n1\n", encoding="utf-8")
    base = {
        "dataset_path": str(data),
        "hash_manifest_path": str(data),
        "lab2_metrics_path": str(data),
        "report_dir": str(tmp_path),
        "checkpoint_path": str(tmp_path / "m.pt"),
    }
    assert dp.DLConfig(**base)  # базовый конфиг валиден
    with pytest.raises(ValidationError):
        dp.DLConfig(**base, lr=-0.1)
    with pytest.raises(ValidationError):
        dp.DLConfig(**base, dropout=1.5)
    with pytest.raises(ValidationError):
        dp.DLConfig(**base, hidden_dims=[])
    with pytest.raises(ValidationError):
        dp.DLConfig(**{**base, "dataset_path": str(tmp_path / "none.csv")})


def test_seed_fixes_weights() -> None:
    """Одинаковое зерно -> идентичные веса MLP; фиксация зерна работает."""
    torch.manual_seed(511)
    m1 = dp.MLP(20, [8, 4], 0.1)
    torch.manual_seed(511)
    m2 = dp.MLP(20, [8, 4], 0.1)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
