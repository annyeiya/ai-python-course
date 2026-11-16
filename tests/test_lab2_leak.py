"""ЛР2, негативные контроли: ловушка утечки и память чанкового режима.

Тест 1: скейлер, обученный на всех данных (утечка), даёт другую трансформацию
теста, чем скейлер, обученный только на train, - значит утечка обнаружима,
и наш конвейер использует только train-статики.
Тест 2: пик памяти чанкового обучения не растёт с удвоением объёма данных.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ml_pipeline as mp


def test_leak_trap_detected() -> None:
    """Ловушка утечки: фит скейлера по всем данным отличим от фита по train."""
    x, y = make_classification(
        n_samples=4000, n_features=20, n_informative=10, n_redundant=5,
        n_repeated=5, random_state=511,
    )
    x_tr, x_te, y_tr, _ = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=511
    )
    s_train = StandardScaler().fit(x_tr)
    s_all = StandardScaler().fit(np.vstack([x_tr, x_te]))
    t_train = s_train.transform(x_te)
    t_all = s_all.transform(x_te)
    # Утечка обнаружима: трансформации теста различаются.
    assert np.abs(t_train - t_all).max() > 1e-6
    # Наш конвейер после fit на train хранит train-статики, а не all-статики.
    pipe = mp.build_full_pipeline(0.1, seed=511)
    pipe.fit(x_tr, y_tr)
    scaler = pipe.named_steps["scaler"]
    assert np.allclose(scaler.mean_, s_train.mean_)
    assert not np.allclose(scaler.mean_, s_all.mean_)


def test_chunks_memory_does_not_grow(tmp_path: Path) -> None:
    """Пик памяти чанкового обучения не растёт с удвоением объёма данных."""
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
        cfg = mp.MLConfig(
            dataset_path=str(path),
            hash_manifest_path=str(path),
            report_dir=str(tmp_path),
            model_path=str(tmp_path / "m.joblib"),
            chunk_size=500,
            n_repeats=1,
        )
        mask = np.ones(n, dtype=bool)
        _, peak, _, _, _ = mp.run_mode(
            cfg, "chunks", 0.1, n, mask, None, y, feats
        )
        peaks.append(peak)
    chunk_mb = 500 * 21 * 8 / 1e6
    assert peaks[1] - peaks[0] < 2 * chunk_mb
