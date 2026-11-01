"""Проверки ЛР4: статистики, сохранение и некорректный вход.
1) совпадают ли статистики full/chunks и вердикты после загрузки NPZ;
2) отклоняется ли текст в числовом поле;
3) отклоняются ли неправильные настройки.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validator_model import DataValidator, ValidatorConfig


@pytest.fixture
def setup_validator(tmp_path):
    """Создаём искусственный набор и чекпоинт только для теста."""

    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 3))
    y = np.tile([0, 1], 40)

    features = ["a", "b", "c"]
    csv = tmp_path / "data.csv"

    pd.DataFrame(
        {
            **{feature: x[:, index] for index, feature in enumerate(features)},
            "target": y,
        }
    ).to_csv(csv, index=False)

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    model = tmp_path / "model.pt"

    torch.save(
        {
            "state_dict": {
                "net.0.weight": torch.ones((1, 3)),
                "net.0.bias": torch.zeros(1),
            },
            "scaler_mean": torch.zeros(3),
            "scaler_scale": torch.ones(3),
            "meta": {
                "n_feats": 3,
                "hidden_dims": [],  # сеть из одного слоя, скрытых нет
                "dropout": 0.0,
            }
        },
        model,
    )

    cfg = ValidatorConfig(
        dataset_path=str(csv),
        hash_manifest_path=str(manifest),
        working_model_path=str(model),
        report_dir=str(tmp_path),
        features=features,
        report_date="2026-11-01",
        chunk_size=7,
    )

    mask = np.ones(len(y), dtype=bool)

    return cfg, x, y, mask


def test_full_chunks_and_reload(setup_validator, tmp_path):
    """Шаги 3 и 5: сравниваем статистики и вердикты после загрузки."""

    cfg, x, y, mask = setup_validator

    full = DataValidator(cfg).fit(mask, "full").calibrate(x)
    chunks = DataValidator(cfg).fit(mask, "chunks").calibrate(x)

    for key in ("mu", "cov", "lo", "hi", "count"):
        assert np.allclose(
            full.stats[key],
            chunks.stats[key],
            atol=1e-6,
            rtol=0,
        )

    assert abs(full.threshold - chunks.threshold) < 1e-6

    path = tmp_path / "validator.npz"
    full.save(path)

    restored = DataValidator(cfg).load(path)

    assert np.array_equal(
        full.decide(x, y)[0],
        restored.decide(x, y)[0],
    )

    assert np.array_equal(
        full.decide(x)[0],
        restored.decide(x)[0],
    )


def test_invalid_category_and_config(setup_validator):
    """Шаги 1, 2 и 4: отклоняем плохие настройки и текст в числе."""

    cfg, x, _, mask = setup_validator

    model = DataValidator(cfg).fit(mask, "chunks").calibrate(x)

    bad = x[:1].astype(object)
    bad[0, 0] = "чужая_категория"

    with pytest.raises(ValueError):
        model.decide(bad)

    with pytest.raises(ValidationError):
        ValidatorConfig.model_validate(
            {**cfg.model_dump(), "chunk_size": 0}
        )

    with pytest.raises(ValidationError):
        ValidatorConfig.model_validate(
            {**cfg.model_dump(), "conf_threshold": 2}
        )
