"""ЛР6: создаём реестр из сохранённых моделей и CSV с метриками."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

METRIC_NAMES = {"accuracy", "pr_auc", "roc_auc"}


def sha256(path):
    """Считаем SHA-256 модели по блокам."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_metrics(path):
    """Читаем по одному значению каждой метрики для режима full."""
    frame = pd.read_csv(path)
    required_columns = {"mode", "metric", "value"}

    if not required_columns.issubset(frame.columns):
        raise ValueError(
            f"В {path} нужны столбцы mode, metric, value"
        )

    subset = frame[
        (frame["mode"] == "full")
        & frame["metric"].isin(METRIC_NAMES)
    ]

    if len(subset) != len(METRIC_NAMES) or set(subset["metric"]) != METRIC_NAMES:
        raise ValueError(
            f"В {path} нужна ровно одна строка full для каждой метрики: "
            "accuracy, pr_auc, roc_auc"
        )

    metrics = {
        row.metric: float(row.value)
        for row in subset.itertuples()
    }

    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"В {path} есть неконечное значение метрики")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="reports/LAB6/registry.json",
    )
    args = parser.parse_args()

    path = Path(args.registry)

    # Не перезаписываем существующую историю.
    if path.exists():
        raise SystemExit(
            f"Реестр уже существует: {path}; не перезаписываем историю"
        )

    now = datetime.now(UTC).isoformat()
    versions = {}

    inputs = [
        (
            "ml-v1",
            "sklearn",
            "reports/LAB2/ml_model.joblib",
            "reports/LAB2/ml_metrics.csv",
            "рабочая",
        ),
        (
            "dl-v1",
            "torch",
            "reports/LAB3/dl_model.pt",
            "reports/LAB3/dl_metrics.csv",
            "пробная",
        ),
    ]

    for version, kind, model_path, metrics_path, status in inputs:
        # Шаг 3: фиксируем хеш файла и читаем метрики из CSV.
        digest = sha256(model_path)
        metrics = read_metrics(metrics_path)

        versions[version] = {
            "kind": kind,
            "path": model_path,
            "sha256": digest,
            "metrics": metrics,
            "status": status,
            "history": [
                {
                    "at": now,
                    "status": "готовится",
                    "reason": "регистрация артефакта модели",
                },
                {
                    "at": now,
                    "status": "пробная",
                    "reason": (
                        "хеш зафиксирован в реестре, "
                        "метрики прочитаны из CSV"
                    ),
                },
            ],
        }

        if status == "рабочая":
            versions[version]["history"].append(
                {
                    "at": now,
                    "status": "рабочая",
                    "reason": "назначена начальной рабочей версией",
                }
            )

    registry = {
        "active": "ml-v1",
        "previous": None,
        "versions": versions,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    # Режим x не позволяет перезаписать существующий файл.
    with path.open("x", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print(f"[реестр] создан {path}; активная версия: ml-v1")


if __name__ == "__main__":
    main()
