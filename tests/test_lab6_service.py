"""ЛР6: проверки схемы, хеша и предупреждения валидатора."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from model_service import PredictRequest, Registry


def test_contract_schema():
    """Шаг 1: проверяем размер, тип и диапазон признаков."""
    request = PredictRequest(
        request_id="ok",
        features=[0.0] * 20,
    )

    assert len(request.features) == 20

    invalid_payloads = [
        {"request_id": "a"},
        {
            "request_id": "a",
            "features": [0] * 19,
        },
        {
            "request_id": "a",
            "features": ["wrong"] + [0] * 19,
        },
        {
            "request_id": "a",
            "features": [1001] + [0] * 19,
        },
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            PredictRequest.model_validate(payload)


def test_bad_hash_does_not_switch(tmp_path):
    """Шаг 3: плохой хеш не меняет модель или файл реестра."""
    model = DummyClassifier(strategy="prior").fit(
        np.zeros((4, 20)),
        [0, 1, 0, 1],
    )

    file = tmp_path / "model.joblib"
    joblib.dump(model, file)

    digest = hashlib.sha256(
        file.read_bytes()
    ).hexdigest()

    entries = {
        "good": {
            "kind": "sklearn",
            "path": str(file),
            "sha256": digest,
            "status": "рабочая",
            "history": [],
        },
        "bad": {
            "kind": "sklearn",
            "path": str(file),
            "sha256": "0" * 64,
            "status": "пробная",
            "history": [],
        },
    }

    path = tmp_path / "registry.json"

    path.write_text(
        json.dumps(
            {
                "active": "good",
                "previous": None,
                "versions": entries,
            }
        ),
        encoding="utf-8",
    )

    registry = Registry(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="SHA-256"):
        registry.switch(
            "bad",
            "негативный контроль",
        )

    assert registry.snapshot()[0] == "good"
    assert path.read_bytes() == before


@pytest.mark.parametrize("suspicious", [False, True])
def test_warning_does_not_block_prediction(monkeypatch, suspicious):
    """Предупреждение не заменяет результат прогноза."""
    import model_service

    class StubRegistry:
        def snapshot(self):
            return "ml-v1", object()

    class StubValidator:
        def decide(self, x):
            assert x.shape == (1, 20)

            return (
                np.array([suspicious]),
                np.array([1.5 if suspicious else 0.5]),
            )

    monkeypatch.setattr(
        model_service,
        "registry",
        StubRegistry(),
    )

    monkeypatch.setattr(
        model_service,
        "validator",
        StubValidator(),
    )

    monkeypatch.setattr(
        model_service,
        "predict",
        lambda model, features: 0.8,
    )

    result = model_service.predict_endpoint(
        PredictRequest(
            request_id="check",
            features=[0.0] * 20,
        )
    )

    assert result.request_id == "check"
    assert result.prediction == 1
    assert result.validation.suspicious is suspicious
    assert result.validation.score == (
        1.5 if suspicious else 0.5
    )
