"""ЛР4: запускаем два валидатора, проверки и создаём артефакты."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from validator_model import DataValidator, ValidatorConfig, sha256


def split_indices(cfg):
    """Проверяем целостность данных и восстанавливаем разбиение ЛР3."""

    manifest = json.loads(
        Path(cfg.hash_manifest_path).read_text(encoding="utf-8")
    )

    entries = [
        entry
        for entry in manifest["files"]
        if Path(entry["path"]) == Path(cfg.dataset_path)
    ]

    data_hash = sha256(cfg.dataset_path)

    if len(entries) != 1 or entries[0]["sha256"] != data_hash:
        raise ValueError("Хеш CSV не совпадает с манифестом ЛР1")

    columns = list(pd.read_csv(cfg.dataset_path, nrows=0).columns)

    if [column for column in columns if column != "target"] != cfg.features:
        raise ValueError(
            "Состав или порядок признаков отличается от CSV"
        )

    y = pd.read_csv(
        cfg.dataset_path,
        usecols=["target"],
    )["target"].to_numpy()

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Ожидается бинарная цель 0/1")

    train_val, test = train_test_split(
        np.arange(len(y)),
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.seed,
    )

    train, val = train_test_split(
        train_val,
        test_size=cfg.val_size / (1 - cfg.test_size),
        stratify=y[train_val],
        random_state=cfg.seed,
    )

    return y, train, val, test, data_hash


def read_rows(cfg, indices, n_rows):
    """Читаем нужные строки порциями и сохраняем порядок indices."""

    mask = np.zeros(n_rows, dtype=bool)
    mask[indices] = True

    pieces = []

    for block in pd.read_csv(
        cfg.dataset_path,
        usecols=cfg.features,
        chunksize=cfg.chunk_size,
    ):
        chosen = mask[block.index.to_numpy()]

        if chosen.any():
            pieces.append(block.loc[chosen])

    return (
        pd.concat(pieces)
        .loc[indices, cfg.features]
        .to_numpy(dtype=np.float64)
    )


# Шаг 4: создаём контроли без отбора по правильности модели.
def make_cases(cfg, x_test, y_test, scale):
    """Создаём свои данные, сдвиг, чужие числа, выброс и переворот."""

    n = min(cfg.alien_size, len(x_test))
    base = x_test[:n]
    labels = y_test[:n]

    rng = np.random.default_rng(cfg.seed)

    outliers = base.copy()
    outliers[:, 0] += cfg.outlier_mult * scale[0]

    return [
        ("own", x_test, y_test, 0),
        (
            "mean_shift",
            base + cfg.shift_sigma * scale,
            labels,
            1,
        ),
        (
            "foreign_numeric",
            rng.uniform(
                -cfg.foreign_scale,
                cfg.foreign_scale,
                base.shape,
            ),
            labels,
            1,
        ),
        ("outlier", outliers, labels, 1),
        ("label_flip", base, 1 - labels, 1),
    ]


# Шаг 4: метрики для числовых примеров.
# Чужую категорию проверяем отдельно как некорректный тип.
def evaluate(model, cases):
    """Считаем precision, recall, accuracy, ложные отказы и ROC-AUC."""

    decisions = []
    scores = []
    truth = []
    rows = []

    for name, x, declared, expected in cases:
        rejected, score = model.decide(x, declared)

        decisions.extend(rejected.tolist())
        scores.extend(score.tolist())
        truth.extend([expected] * len(x))

        rows.extend(
            {
                "case": name,
                "row": index,
                "expected_reject": expected,
                "actual_reject": int(verdict),
            }
            for index, verdict in enumerate(rejected)
        )

    pred = np.asarray(decisions, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())

    metrics = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": float(np.mean(pred == truth)),
        "recall": tp / int(truth.sum()),
        "false_reject_rate": fp / int((~truth).sum()),
        "roc_auc": float(roc_auc_score(truth, scores)),
    }

    return metrics, pd.DataFrame(rows)


def write_report(
    cfg,
    out,
    results,
    max_diff,
    threshold_diff,
    modes_equal,
    repeat_ok,
    data_hash,
    model_hash,
    params_hash,
    negative,
    silent,
):
    """Записываем только фактические результаты текущего запуска."""

    lines = [
        "# ЛР4. Модель-валидатор, вариант 10, профиль P1",
        "",
        f"Автор: {cfg.author}; дата: {cfg.report_date}",
        "",
        "Коммит кода: ",
        "",
        f"SHA256 данных: {data_hash}",
        "",
        f"SHA256 рабочей модели: {model_hash}",
        "",
        f"SHA256 validator_params.npz: {params_hash}",
        "",
        "## Шаг 1. Схема настроек",

        f"Метод: {cfg.proximity_method}; ",
        "",
        f"признаков: {len(cfg.features)}; ",
        "",
        f"размер чанка: {cfg.chunk_size}.",
        "",

        f"Квантиль val: {cfg.dist_quantile}; ",
        "",
        f"порог уверенности: {cfg.conf_threshold}.",
        "",

        f"Требования: recall >= {cfg.recall_min}; "
        f"false_reject_rate <= {cfg.false_reject_max}.",

        "",
        "## Шаг 2. Правило отказа",

        "Средние двух классов, общая внутриклассовая "
        "ковариация с регуляризацией и диапазоны вычислены на train.",
        "",

        "Отказ: расстояние Махаланобиса до ближайшего класса "
        "выше порога ИЛИ уверенность модели ниже порога.",
        "",

        "Диапазоны сохранены как статистики; "
        "отдельного отказа по диапазону нет.",
        "",

        "Если метка передана, уверенность — вероятность этой "
        "метки; без метки — вероятность предсказанного класса.",
        "",

        "Масштабирование и веса взяты из ЛР3; порог расстояния "
        "калиброван на val, тест не используется при подборе.",

        "",
        "## Шаг 3. Сравнение статистик",

        f"Максимальная разница: {max_diff:.12g}; ",
        f"допуск: {cfg.stat_tol}; "
        f"результат: {max_diff <= cfg.stat_tol}.",
        "",
        f"Разница порогов: {threshold_diff:.12g}; "
        f"совпадение поштучных вердиктов режимов: {modes_equal}.",
        "",
        "Только fit(chunks) читает признаки порциями; "
        "индексы и оценочные выборки хранятся в памяти.",
        "",
        "## Шаг 4. Метрики отказа",

            "mean_shift - Сдвинули числовые признаки - Отказ",
            "",
            "foreign_numeric - Подали числа из другого диапазона - Отказ",
            "",
            "outlier - Добавили большой выброс - Отказ",
            "",
            "label_flip - Поменяли 0 на 1 и наоборот - Отказ, если валидатор обнаружит несогласие",
            "",
            "foreign_category - Подали текст в числовое поле - Ошибка проверки входа",
            "",
            "own - Оставили свои тестовые данные - Приём, кроме допустимой доли ложных отказов",
            "",
        (
            "| Режим | Precision | Accuracy | Recall | "
            "Ложные отказы | ROC-AUC | Run ID |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for mode in ("full", "chunks"):
        values = results[mode]["metrics"]
        run_id = results[mode]["run_id"]

        lines.append(
            f"| {mode} | {values['precision']:.6f} | "
            f"{values['accuracy']:.6f} | "
            f"{values['recall']:.6f} | "
            f"{values['false_reject_rate']:.6f} | "
            f"{values['roc_auc']:.6f} | {run_id} |"
        )

    lines += [
        "",
        "| Режим | Контроль | Всего | Отказано |",
        "|---|---|---:|---:|",
    ]

    grouped = negative.groupby(
        ["mode", "case"],
        sort=False,
    )

    for (mode, name), group in grouped:
        lines.append(
            f"| {mode} | {name} | {len(group)} | "
            f"{group['actual_reject'].sum()} |"
        )

    for mode in ("full", "chunks"):
        values = results[mode]["metrics"]

        lines.append(
            f"{mode}: recall >= порога: "
            f"{values['recall'] >= cfg.recall_min}; "
            f"ложные отказы <= порога: "
            f"{values['false_reject_rate'] <= cfg.false_reject_max}."
        )

    lines += [
        (
            f"Чужих входов без отказа в обоих режимах: {silent}; "
            f"критерий нуля: {silent == 0}."
        ),
        (
            "Все выбранные метки перевёрнуты без отбора по "
            "правильности модели; часть может не обнаруживаться."
            ""
        ),
        (
            "Чужая категория — отдельно проверенный некорректный тип, "
            "а не непрерывное числовое распределение."
            ""
        ),
        "",
        "## Шаг 5. Артефакты и воспроизводимость",
        (
            f"Хеш NPZ полного режима: {params_hash}; "
            f"поштучная проверка после загрузки обоих NPZ: {repeat_ok}."
            ""
        ),
        (
            "validator_metrics.csv — результаты и ID прогонов; "
            "validator_negative.log — ожидаемые и фактические вердикты."
            ""
        ),
        (
            "ROC-AUC вычислен на числовых примерах, с учётом равных "
            "оценок. Метрики зависят от состава контрольного набора."
            ""
        ),
        "",
    ]

    (out / "lab4_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Шаг 1: проверяем YAML схемой Pydantic.
    raw_config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )

    cfg = ValidatorConfig.model_validate(raw_config)

    out = Path(cfg.report_dir)
    out.mkdir(parents=True, exist_ok=True)

    y, train, val, test, data_hash = split_indices(cfg)

    mask = np.zeros(len(y), dtype=bool)
    mask[train] = True

    x_val = read_rows(cfg, val, len(y))
    x_test = read_rows(cfg, test, len(y))

    # Шаг 3: независимо обучаем и калибруем два экземпляра класса.
    models = {
        mode: DataValidator(cfg).fit(mask, mode).calibrate(x_val)
        for mode in ("full", "chunks")
    }

    max_diff = max(
        float(
            np.max(
                np.abs(
                    models["full"].stats[key]
                    - models["chunks"].stats[key]
                )
            )
        )
        for key in ("mu", "cov", "lo", "hi", "count")
    )

    threshold_diff = abs(
        models["full"].threshold - models["chunks"].threshold
    )

    cases = make_cases(
        cfg,
        x_test,
        y[test],
        models["full"].scale,
    )

    results = {}
    frames = []
    repeat_ok = True

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("lab4-variant10")

    for mode, model in models.items():
        # Шаг 4: сравниваем отказы с ожидаемыми вердиктами.
        metrics, records = evaluate(model, cases)

        # Шаг 5: сохраняем NPZ и проверяем загрузку по каждому примеру.
        path = out / (
            "validator_params.npz"
            if mode == "full"
            else "validator_params_chunks.npz"
        )

        digest = model.save(path)

        restored = DataValidator(cfg).load(path)
        _, check = evaluate(restored, cases)

        repeat_ok = repeat_ok and records["actual_reject"].equals(
            check["actual_reject"]
        )

        # Шаг 4: текст в числовом поле должен быть отклонён.
        try:
            bad = x_test[:1].astype(object)
            bad[0, 0] = "чужая_категория"

            model.decide(bad)
            rejected = 0

        except (ValueError, TypeError):
            rejected = 1

        records.loc[len(records)] = [
            "foreign_category",
            0,
            1,
            rejected,
        ]

        records.insert(0, "mode", mode)
        frames.append(records)

        # Шаг 5: регистрируем параметры, метрики и NPZ в MLflow.
        with mlflow.start_run(run_name=f"validator-{mode}") as run:
            mlflow.log_params(
                {
                    "mode": mode,
                    "seed": cfg.seed,
                    "chunk_size": cfg.chunk_size,
                    "distance_quantile": cfg.dist_quantile,
                    "threshold": model.threshold,
                    "conf_threshold": model.conf,
                    "params_sha256": digest,
                }
            )

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(path))
            run_id = run.info.run_id

        results[mode] = {
            "metrics": metrics,
            "run_id": run_id,
            "hash": digest,
            "records": records,
            "threshold": model.threshold,
        }

    modes_equal = results["full"]["records"]["actual_reject"].equals(
        results["chunks"]["records"]["actual_reject"]
    )

    negative = pd.concat(frames, ignore_index=True)

    negative["verdict"] = np.where(
        negative.expected_reject == negative.actual_reject,
        "совпал",
        "НЕ совпал",
    )

    negative.to_csv(
        out / "validator_negative.log",
        index=False,
    )

    silent = int(
        (
            (negative.expected_reject == 1)
            & (negative.actual_reject == 0)
        ).sum()
    )

    table = [
        {
            "mode": mode,
            "run_id": result["run_id"],
            **result["metrics"],
            "stat_max_diff": max_diff,
            "threshold": result["threshold"],
            "threshold_diff": threshold_diff,
        }
        for mode, result in results.items()
    ]

    pd.DataFrame(table).to_csv(
        out / "validator_metrics.csv",
        index=False,
    )

    write_report(
        cfg,
        out,
        results,
        max_diff,
        threshold_diff,
        modes_equal,
        repeat_ok,
        data_hash,
        sha256(cfg.working_model_path),
        results["full"]["hash"],
        negative,
        silent,
    )

    ok = (
        max_diff <= cfg.stat_tol
        and threshold_diff <= cfg.stat_tol
        and repeat_ok
        and all(
            result["metrics"]["recall"] >= cfg.recall_min
            and result["metrics"]["false_reject_rate"]
            <= cfg.false_reject_max
            for result in results.values()
        )
        and silent == 0
    )

    print(
        f"Статистики: разница={max_diff:.3g}; "
        f"пороги: разница={threshold_diff:.3g}; "
        f"повтор={repeat_ok}"
    )

    print(
        f"Чужих входов без отказа в обоих режимах: {silent}. "
        f"Артефакты: {out}"
    )

    print(
        "Критерии всех проверок: "
        f"{'ВЫПОЛНЕНЫ' if ok else 'НЕ ВЫПОЛНЕНЫ'}"
    )

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
