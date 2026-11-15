"""ЛР5: тесты схемы, режимов, дрейфа и воспроизводимости."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generator_model import GeneratorConfig, SyntheticGenerator


@pytest.fixture
def reference():
    """Небольшая таблица с двумя классами и известными границами."""

    return pd.DataFrame(
        {
            "a": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )


def config(**changes):
    """Готовим настройки для каждого независимого теста."""

    base = {
        "mode": "similar",
        "n_rows": 500,
        "seed": 511,
        "features": ["a", "b"],
        "target": "target",
    }

    return GeneratorConfig.model_validate(
        {**base, **changes}
    )


def test_schema_and_bounds(reference):
    """Шаги 1–2: размеры, столбцы, классы и диапазоны."""

    for mode in ("similar", "random"):
        result = SyntheticGenerator(
            config(mode=mode),
            reference,
        ).generate()

        assert list(result.columns) == ["a", "b", "target"]
        assert len(result) == 500
        assert set(result.target.unique()) <= {0, 1}

        for column in ("a", "b"):
            assert result[column].between(
                reference[column].min(),
                reference[column].max(),
            ).all()


def test_mean_shift(reference):
    """Шаг 2: положительный сдвиг увеличивает среднее."""

    original = SyntheticGenerator(
        config(),
        reference,
    ).generate()

    changed = SyntheticGenerator(
        config(drift={"mean_shift": 1}),
        reference,
    ).generate()

    assert (
        changed[["a", "b"]].mean()
        > original[["a", "b"]].mean()
    ).all()

    assert changed.a.max() <= reference.a.max()
    assert (changed.target == original.target).all()


def test_class_share(reference):
    """Шаг 2: доля класса управляет генерацией до переворота."""

    zero = SyntheticGenerator(
        config(drift={"class_one_share": 0}),
        reference,
    ).generate()

    one = SyntheticGenerator(
        config(drift={"class_one_share": 1}),
        reference,
    ).generate()

    assert (zero.target == 0).all()
    assert (one.target == 1).all()

    assert zero.a.max() <= 0
    assert one.a.min() >= 1


def test_label_flip(reference):
    """Шаг 2: вероятность 1 переворачивает все метки."""

    base = SyntheticGenerator(
        config(),
        reference,
    ).generate()

    flipped = SyntheticGenerator(
        config(drift={"label_flip": 1}),
        reference,
    ).generate()

    pd.testing.assert_frame_equal(
        base[["a", "b"]],
        flipped[["a", "b"]],
    )

    assert (
        flipped.target.to_numpy()
        == 1 - base.target.to_numpy()
    ).all()


def test_season(reference):
    """Шаг 2: синус с периодом четыре меняет заданные фазы."""

    model = SyntheticGenerator(
        config(
            drift={
                "season_amplitude": 0.2,
                "season_period": 4,
            }
        ),
        reference,
    )

    base = np.tile([0.5, 3.5], (4, 1))
    changed = model._apply_drift(base)

    assert np.allclose(changed[0], base[0])
    assert (changed[1] > base[1]).all()
    assert np.allclose(changed[2], base[2])
    assert (changed[3] < base[3]).all()


def test_reproducibility_and_modes(reference, tmp_path):
    """Шаги 2 и 4: проверяем байты, зерно и отличие режимов."""

    model = SyntheticGenerator(config(), reference)
    first = model.csv_bytes()

    assert first == model.csv_bytes()

    path = model.save_csv(tmp_path / "out" / "sample.csv")
    assert path.read_bytes() == first

    assert first != SyntheticGenerator(
        config(seed=512),
        reference,
    ).csv_bytes()

    assert first != SyntheticGenerator(
        config(mode="random"),
        reference,
    ).csv_bytes()


def test_invalid_config():
    """Шаг 1: схема отвергает некорректные настройки."""

    bad_changes = (
        {"mode": "other"},
        {"n_rows": 0},
        {"seed": -1},
        {"features": ["a", "a"]},
        {"features": ["a", "target"]},
        {"features": [" "]},
        {"drift": {"label_flip": 1.1}},
        {"drift": {"class_one_share": -1}},
        {"drift": {"season_period": 1}},
    )

    for change in bad_changes:
        with pytest.raises(ValidationError):
            config(**change)


def test_invalid_reference(reference):
    """Шаг 2: отклоняем неверную схему, NaN и один класс."""

    with pytest.raises(ValueError):
        SyntheticGenerator(
            config(),
            reference.drop(columns="b"),
        )

    with pytest.raises(ValueError):
        SyntheticGenerator(
            config(),
            reference.iloc[0:0],
        )

    bad = reference.copy()
    bad.loc[0, "a"] = np.nan

    with pytest.raises(ValueError):
        SyntheticGenerator(config(), bad)

    with pytest.raises(ValueError):
        SyntheticGenerator(
            config(),
            reference.assign(target=0),
        )
