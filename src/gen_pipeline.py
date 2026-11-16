"""ЛР5: эксперимент с генератором и создание отчёта по двум режимам."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from generator_model import GeneratorConfig, SyntheticGenerator


def sha256(path):
    """Считаем хеш входного файла частями."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_real(path, manifest_path, cfg, split_seed):
    """Проверяем манифест и делим данные."""

    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )

    match = [
        entry
        for entry in manifest["files"]
        if Path(entry["path"]) == Path(path)
    ]

    if len(match) != 1 or match[0]["sha256"] != sha256(path):
        raise ValueError("Хеш датасета не совпадает с манифестом")

    frame = pd.read_csv(path)

    if list(frame.columns) != cfg.features + [cfg.target]:
        raise ValueError(
            "Состав или порядок столбцов отличается от схемы выхода"
        )

    if set(frame[cfg.target].unique()) != {0, 1}:
        raise ValueError("Цель должна содержать классы 0 и 1")

    indices = np.arange(len(frame))

    train_val, test = train_test_split(
        indices,
        test_size=0.15,
        stratify=frame[cfg.target],
        random_state=split_seed,
    )

    train, _ = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=frame[cfg.target].to_numpy()[train_val],
        random_state=split_seed,
    )

    return (
        frame.iloc[train].reset_index(drop=True),
        frame.iloc[test].reset_index(drop=True),
    )


# PSI по квантильным корзинам референса.
def psi(reference, actual, bins=10):
    """Чем выше PSI, тем сильнее различаются распределения."""

    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(edges) < 2:
        return (
            0.0
            if np.all(actual == reference[0])
            else float("inf")
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref = np.histogram(reference, bins=edges)[0].astype(float)
    gen = np.histogram(actual, bins=edges)[0].astype(float)

    # Сглаживание предотвращает деление на ноль.
    # После добавления псевдосчётчика доли нормируются заново.
    ref = (ref + 0.5) / (ref.sum() + 0.5 * len(ref))
    gen = (gen + 0.5) / (gen.sum() + 0.5 * len(gen))

    return float(
        np.sum((gen - ref) * np.log(gen / ref))
    )


def compare(real, samples, cfg):
    """Собираем средние, PSI и KS для каждого признака."""

    rows = []

    for feature in cfg.features:
        ref = real[feature].to_numpy(dtype=float)

        for mode, frame in samples.items():
            values = frame[feature].to_numpy(dtype=float)

            rows.append(
                {
                    "feature": feature,
                    "mode": mode,
                    "real_mean": float(ref.mean()),
                    "generated_mean": float(values.mean()),
                    "psi": psi(ref, values),
                    "ks": float(ks_2samp(ref, values).statistic),
                }
            )

    return pd.DataFrame(rows)

# Шаг 2: дополнительные независимые вызовы генератора с разными ручками.
def run_drift_experiments(cfg, train):
    """Сравниваем каждый дрейф с нулевым дрейфом при том же зерне."""

    # Для каждого эксперимента остальные ручки остаются нейтральными.
    neutral = {
        "mean_shift": 0.0,
        "class_one_share": None,
        "label_flip": 0.0,
        "season_amplitude": 0.0,
        "season_period": 20,
    }

    def generate_with(changes):
        """Создаём отдельный генератор с одним изменённым параметром."""

        settings = {
            **cfg.model_dump(),
            "mode": "similar",
            "drift": {**neutral, **changes},
        }

        model = SyntheticGenerator(
            GeneratorConfig.model_validate(settings),
            train,
        )

        return model.generate(), model.std

    # Базовый прогон нужен для парного сравнения изменений.
    base, std = generate_with({})

    # Постоянные признаки исключаем из деления на std.
    valid = std > 1e-12

    if not valid.any():
        raise ValueError(
            "Для демонстрации дрейфа нужен хотя бы "
            "один непостоянный признак"
        )

    features = cfg.features
    base_x = base[features].to_numpy(dtype=float)
    base_y = base[cfg.target].to_numpy()

    rows = []

    def add(
        handle,
        value,
        metric,
        observed,
        passed,
        other_period=None,
        other_projection=None,
    ):
        """Добавляем измерение и результат проверки в таблицу."""

        rows.append(
            {
                "handle": handle,
                "value": value,
                "metric": metric,
                "observed": float(observed),
                "other_period": other_period,
                "other_projection": other_projection,
                "passed": bool(passed),
            }
        )

    # Шаг 2: сравниваем средний сдвиг в std с базовым прогоном.
    for value in (-1.0, 0.0, 1.0):
        frame, _ = generate_with({"mean_shift": value})

        delta = (
            frame[features].to_numpy() - base_x
        )[:, valid] / std[valid]

        observed = float(delta.mean())

        if value < 0:
            direction = observed < -1e-12
        elif value > 0:
            direction = observed > 1e-12
        else:
            direction = abs(observed) <= 1e-12

        labels_unchanged = np.array_equal(
            frame[cfg.target].to_numpy(),
            base_y,
        )

        add(
            "mean_shift",
            value,
            "средний сдвиг в std",
            observed,
            direction and labels_unchanged,
        )

    # Шаг 2: проверяем фактическую долю класса с учётом случайного разброса.
    for value in (0.1, 0.5, 0.9):
        frame, _ = generate_with({"class_one_share": value})

        observed = float(
            (frame[cfg.target] == 1).mean()
        )

        tolerance = (
            4 * np.sqrt(value * (1 - value) / len(frame))
            + 1 / len(frame)
        )

        add(
            "class_one_share",
            value,
            "фактическая доля класса 1",
            observed,
            abs(observed - value) <= tolerance,
        )

    # Шаг 2: сравниваем метки с базовым прогоном.
    # Дополнительно проверяем, что признаки не изменились.
    for value in (0.0, 0.25, 1.0):
        frame, _ = generate_with({"label_flip": value})

        observed = float(
            np.mean(frame[cfg.target].to_numpy() != base_y)
        )

        tolerance = (
            4 * np.sqrt(value * (1 - value) / len(frame))
            + 1 / len(frame)
        )

        features_unchanged = np.array_equal(
            frame[features].to_numpy(),
            base_x,
        )

        add(
            "label_flip",
            value,
            "доля изменённых меток",
            observed,
            features_unchanged
            and abs(observed - value) <= tolerance,
        )

    # Шаг 2: измеряем эффект сезонной волны относительно базы.
    period = 20

    phase = np.sin(
        2 * np.pi * np.arange(len(base)) / period
    )

    for value in (0.0, 0.5, 1.0):
        frame, _ = generate_with(
            {
                "season_amplitude": value,
                "season_period": period,
            }
        )

        delta = (
            frame[features].to_numpy() - base_x
        )[:, valid] / std[valid]

        observed = float(
            np.mean(delta * phase[:, None])
        )

        if value == 0:
            direction = abs(observed) <= 1e-12
        else:
            direction = observed > 1e-12

        labels_unchanged = np.array_equal(
            frame[cfg.target].to_numpy(),
            base_y,
        )

        add(
            "season_amplitude",
            value,
            "проекция сдвига на синус, std",
            observed,
            direction and labels_unchanged,
        )

    # Шаг 2: меняем период и сравниваем с другой частотой.
    for period in (10, 20, 40):
        frame, _ = generate_with(
            {
                "season_amplitude": 0.5,
                "season_period": period,
            }
        )

        delta = (
            frame[features].to_numpy() - base_x
        )[:, valid] / std[valid]

        signal = np.sin(
            2 * np.pi * np.arange(len(base)) / period
        )

        other_period = 20 if period != 20 else 10

        other = np.sin(
            2 * np.pi * np.arange(len(base)) / other_period
        )

        observed = float(
            np.mean(delta * signal[:, None])
        )

        other_projection = float(
            np.mean(delta * other[:, None])
        )

        labels_unchanged = np.array_equal(
            frame[cfg.target].to_numpy(),
            base_y,
        )

        add(
            "season_period",
            period,
            "проекция на синус заданного периода",
            observed,
            observed > abs(other_projection) and labels_unchanged,
            other_period,
            other_projection,
        )

    return pd.DataFrame(rows)

# Шаги 2–3: демонстрируем ручки и сравниваем результат с реальными данными.
def run_drift_experiments_2(cfg, train, real_test):
    """Возвращаем таблицы признаков и долей для сценариев без вердиктов."""

    # В каждом сценарии меняем указанную ручку, остальные нейтральны.
    neutral = {
        "mean_shift": 0.0,
        "class_one_share": None,
        "label_flip": 0.0,
        "season_amplitude": 0.0,
        "season_period": 20,
    }

    scenarios = [
        (
            "mean_shift",
            -1.0,
            {"mean_shift": -1.0},
        )
    ] + [
        (
            "class_one_share",
            0.1,
            {"class_one_share": 0.1},
        )
    ] + [
        (
            "label_flip",
            0.25,
            {"label_flip": 0.25},
        )
    ] + [
        (
            "season_amplitude",
            0.5,
            {"season_amplitude": 0.5},
        )
    ] + [
        (
            "season_period",
            20,
            {
                "season_amplitude": 0.5,
                "season_period": 20,
            },
        )
    ]

    feature_rows = []
    class_rows = []

    real_share = float(
        (real_test[cfg.target] == 1).mean()
    )

    # Базовые метки нужны, чтобы увидеть переворот даже
    # при почти равных долях классов.
    baseline_labels = {}

    for mode in ("similar", "random"):
        settings = {
            **cfg.model_dump(),
            "mode": mode,
            "drift": neutral,
        }

        model = SyntheticGenerator(
            GeneratorConfig.model_validate(settings),
            train,
        )

        baseline_labels[mode] = (
            model.generate()[cfg.target].to_numpy()
        )

    for handle, value, changes in scenarios:
        # Шаг 2: создаём оба режима с одним зерном
        # и выбранными настройками дрейфа.
        samples = {}

        for mode in ("similar", "random"):
            settings = {
                **cfg.model_dump(),
                "mode": mode,
                "drift": {**neutral, **changes},
            }

            model = SyntheticGenerator(
                GeneratorConfig.model_validate(settings),
                train,
            )

            samples[mode] = pd.read_csv(
                io.BytesIO(model.csv_bytes())
            )

        # Шаг 3: средние, PSI и KS каждого признака
        # относительно отложенной реальной test-выборки.
        table = compare(real_test, samples, cfg)

        similar = (
            table[table["mode"] == "similar"]
            .set_index("feature")
        )

        random = (
            table[table["mode"] == "random"]
            .set_index("feature")
        )

        for feature in cfg.features:
            similar_row = similar.loc[feature]
            random_row = random.loc[feature]

            feature_rows.append(
                {
                    "handle": handle,
                    "value": value,
                    "feature": feature,
                    "real_mean": similar_row["real_mean"],
                    "similar_mean": similar_row["generated_mean"],
                    "random_mean": random_row["generated_mean"],
                    "psi_similar": similar_row["psi"],
                    "ks_similar": similar_row["ks"],
                    "psi_random": random_row["psi"],
                    "ks_random": random_row["ks"],
                }
            )

        # Переворот меток не меняет признаки.
        # Поэтому отдельно сохраняем доли классов и изменённых меток.
        changed = {}

        for mode in ("similar", "random"):
            if handle == "label_flip":
                changed[mode] = float(
                    np.mean(
                        samples[mode][cfg.target].to_numpy()
                        != baseline_labels[mode]
                    )
                )
            else:
                changed[mode] = None

        class_rows.append(
            {
                "handle": handle,
                "value": value,
                "real_class_1": real_share,
                "similar_class_1": float(
                    (samples["similar"][cfg.target] == 1).mean()
                ),
                "random_class_1": float(
                    (samples["random"][cfg.target] == 1).mean()
                ),
                "similar_labels_changed": changed["similar"],
                "random_labels_changed": changed["random"],
            }
        )

    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(class_rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )

    # создаём настройки Pydantic из раздела generator.
    cfg = GeneratorConfig.model_validate(raw["generator"])

    out = Path(raw["report_dir"])
    generated = Path(raw["generated_dir"])

    out.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)

    coverage_path = out / "coverage.json"

    if not coverage_path.is_file():
        raise ValueError(
            "Нет reports/LAB5/coverage.json: "
            "сначала запустите pytest-cov по инструкции"
        )

    coverage = json.loads(
        coverage_path.read_text(encoding="utf-8")
    )

    modules = [
        value
        for key, value in coverage["files"].items()
        if key.replace("\\", "/").endswith(
            "src/generator_model.py"
        )
    ]

    if len(modules) != 1:
        raise ValueError(
            "coverage.json не содержит покрытие src/generator_model.py"
        )

    cov = modules[0]["summary"]

    train, test = load_real(
        raw["dataset_path"],
        raw["hash_manifest_path"],
        cfg,
        raw["split_seed"],
    )

    # Шаг 2: отдельно демонстрируем ручки, не меняя основной эксперимент.
    drift_results = run_drift_experiments(cfg, train)

    # Сохраняем фактические значения и результаты проверок.
    drift_results.to_csv(
        out / "drift_comparison.csv",
        index=False,
        float_format="%.8f",
    )

    # Шаги 2–3: демонстрируем влияние ручек на оба режима.
    drift_features, drift_shares = run_drift_experiments_2(
        cfg,
        train,
        test,
    )

    # Сохраняем фактические значения без проверок «прошло / не прошло».
    drift_features.to_csv(
        out / "drift_feature_comparison.csv",
        index=False,
        float_format="%.8f",
    )

    drift_shares.to_csv(
        out / "drift_class_shares.csv",
        index=False,
        float_format="%.8f",
    )

    samples = {}
    hashes = {}
    repeats = {}

    # для каждого режима создаём независимые экземпляры.
    for mode in ("similar", "random"):
        mode_cfg = GeneratorConfig.model_validate(
            {**cfg.model_dump(), "mode": mode}
        )

        generator = SyntheticGenerator(mode_cfg, train)

        first = generator.csv_bytes()

        second = SyntheticGenerator(
            mode_cfg,
            train,
        ).csv_bytes()

        repeats[mode] = first == second
        hashes[mode] = hashlib.sha256(first).hexdigest()

        path = generated / f"{mode}.csv"
        path.write_bytes(first)

        samples[mode] = pd.read_csv(path)

    # сравниваем генерацию с отложенной test-частью.
    comparison = compare(test, samples, cfg)

    comparison.to_csv(
        out / "feature_comparison.csv",
        index=False,
        float_format="%.8f",
    )

    similar = comparison[comparison["mode"] == "similar"]
    random = comparison[comparison["mode"] == "random"]

    similar_fraction = float(
        (
            (similar["psi"] <= 0.1)
            | (similar["ks"] <= 0.1)
        ).mean()
    )

    random_fraction = float(
        (random["psi"] >= 0.2).mean()
    )

    class_rows = [
        {
            "source": name,
            "class_0": float(
                (frame[cfg.target] == 0).mean()
            ),
            "class_1": float(
                (frame[cfg.target] == 1).mean()
            ),
        }
        for name, frame in {
            "real_test": test,
            **samples,
        }.items()
    ]

    pd.DataFrame(class_rows).to_csv(
        out / "class_shares.csv",
        index=False,
        float_format="%.8f",
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    lines = [
        "# ЛР5. Управляемый генератор, вариант 10, профиль P1",
        "",
        f"Автор: {raw['author']}; дата: {raw['report_date']}",
        "",
        f"Коммит кода: {commit}",
        "",
        f"Хеш входных данных SHA256: {sha256(raw['dataset_path'])}",
        "",
        f"Train: {len(train)} строк; ",
        "",
        f"отложенный test: {len(test)} строк; ",
        "",
        f"генерация: {cfg.n_rows} строк на режим.",
        "",
        "## Шаг 1. Настройки",
        (
            f"Признаки: {', '.join(cfg.features)}; "
            f"цель: {cfg.target}; зерно: {cfg.seed}."
        ),
        "",
        f"Режимы: similar / random; ручки: {cfg.drift.model_dump()}",
        "",
        "## Шаг 2. Генератор",
        (
            "similar: бутстрэп целых обучающих строк отдельно "
            "по классам; random: независимый равномерный шум "
            "в пределах train."
        ),
        "",
        (
            "Сдвиг и сезонная волна измеряются в std train; "
            "результаты ограничены min/max train. Период относится "
            "к номеру синтетической строки, не к реальному времени."
        ),
        "",
        (
            "class_one_share задаёт ожидаемую долю до переворота; "
            "label_flip задаёт вероятность независимого переворота метки."
        ),
        "",
        "## Шаг 3. Сравнение с отложенной реальной выборкой",
        "| Режим | Показатель | Факт | Порог | Вердикт |",
        "|---|---|---:|---:|---|",
        (
            "| similar | доля признаков PSI <= 0.1 ИЛИ KS <= 0.1 | "
            f"{similar_fraction:.4f} | >= 0.8 | "
            f"{'выполнен' if similar_fraction >= 0.8 else 'НЕ выполнен'} |"
        ),
        (
            "| random | доля признаков PSI >= 0.2 | "
            f"{random_fraction:.4f} | >= 0.5 | "
            f"{'выполнен' if random_fraction >= 0.5 else 'НЕ выполнен; см. пояснение'} |"
        ),
        "PSI использует квантильные корзины test и сглаживание 0.5;",
        "",
        "KS — максимальная разница эмпирических функций распределения.",
        "",
        "PSI сравнивает доли значений в корзинах распределения.Здесь корзины строятся по квантилям реальной test-части",
        "",
        "KS — максимальная разница между эмпирическими функциями распределения реальных и сгенерированных значений",
        "",
        (
            "Средние, PSI и KS каждого признака — также в "
            "feature_comparison.csv; доли классов — в class_shares.csv."
        ),
        "",
        (
            "| Признак | Среднее test | Среднее similar | "
            "Среднее random | PSI similar | KS similar | "
            "PSI random | KS random |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for feature in cfg.features:
        a = similar[similar["feature"] == feature].iloc[0]
        b = random[random["feature"] == feature].iloc[0]

        lines.append(
            f"| {feature} | {a['real_mean']:.4f} | "
            f"{a['generated_mean']:.4f} | "
            f"{b['generated_mean']:.4f} | "
            f"{a['psi']:.4f} | {a['ks']:.4f} | "
            f"{b['psi']:.4f} | {b['ks']:.4f} |"
        )

    lines += [
        "",
        "| Источник | Доля класса 0 | Доля класса 1 |",
        "|---|---:|---:|",
    ]

    for row in class_rows:
        lines.append(
            f"| {row['source']} | "
            f"{row['class_0']:.4f} | "
            f"{row['class_1']:.4f} |"
        )

    if random_fraction < 0.5:
        lines.append(
            "Пояснение: независимый равномерный шум имеет те же "
            "границы, что train; близость отдельных маргинальных "
            "распределений допустима, даже если совместная структура "
            "утрачена. Число признаков приведено выше."
        )

    # Шаг 2: показываем в отчёте результаты вызовов с разными ручками.
    lines += [
        "",
        "## Проверка ручек дрейфа",
        (
            "Отдельные вызовы similar, одно зерно; прочие ручки "
            "обнулены. Сдвиги ограничены диапазонами train."
        ),
        "",
        (
            "Доля класса и доля переворотов случайны; допуск равен "
            "четырём стандартным ошибкам плюс 1/n."
        ),
        "",
        (
            "| Ручка | Значение | Измерение | Факт | "
            "Другая частота | Проверка |"
        ),
        "|---|---:|---|---:|---|---|",
    ]

    for _, row in drift_results.iterrows():
        status = (
            "выполнена"
            if row["passed"]
            else "НЕ выполнена"
        )

        if pd.isna(row["other_period"]):
            other = "—"
        else:
            other = (
                f"период {int(row['other_period'])}: "
                f"{row['other_projection']:.6f}"
            )

        lines.append(
            f"| {row['handle']} | {row['value']:.2f} | "
            f"{row['metric']} | {row['observed']:.6f} | "
            f"{other} | {status} |"
        )

    lines += [
        (
            "Для проверки амплитуды период равен 20 строкам; "
            "период отдельно проверяется при амплитуде 0.5. "
            "Проекция — среднее произведение изменения признака "
            "в std на синус фазы; ноль соответствует отсутствию волны."
        ),
        "",
    ]
    # Шаги 2–3: показываем таблицу признаков для каждого сценария.
    lines += [
        "",
        "## Демонстрация ручек дрейфа",
    ]

    for _, scenario in drift_shares.iterrows():
        handle = scenario["handle"]
        value = scenario["value"]

        lines += [
            "",
            f"### {handle} = {value:g}",
            (
                f"Доля класса 1: test={scenario['real_class_1']:.4f}; "
                f"similar={scenario['similar_class_1']:.4f}; "
                f"random={scenario['random_class_1']:.4f}."
            ),
            "",
            (
                "| Признак | Среднее test | Среднее similar | "
                "Среднее random | PSI similar | KS similar | "
                "PSI random | KS random |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]

        subset = drift_features[
            (drift_features["handle"] == handle)
            & (drift_features["value"] == value)
        ]

        for _, row in subset.iterrows():
            lines.append(
                f"| {row['feature']} | "
                f"{row['real_mean']:.4f} | "
                f"{row['similar_mean']:.4f} | "
                f"{row['random_mean']:.4f} | "
                f"{row['psi_similar']:.4f} | "
                f"{row['ks_similar']:.4f} | "
                f"{row['psi_random']:.4f} | "
                f"{row['ks_random']:.4f} |"
            )

        if handle == "label_flip":
            lines += [
                "",
                (
                    "Изменено меток относительно базы: "
                    f"similar={scenario['similar_labels_changed']:.4f}; "
                    f"random={scenario['random_labels_changed']:.4f}."
                ),
            ]

    lines.append(
        "Полные данные: drift_feature_comparison.csv и "
        "drift_class_shares.csv; доля класса 0 равна "
        "1 минус доля класса 1."
    )

    lines += [
        "",
        "## Шаг 4. Побайтовая воспроизводимость (в одной среде)",
    ]

    for mode in ("similar", "random"):
        lines.append(
            f"- {mode}: два независимых вызова равны: "
            f"{repeats[mode]}; SHA256 CSV: {hashes[mode]}."
        )

    lines += [
        "",
        "## Шаг 5. Покрытие pytest-cov",
        "| Модуль | Операторов | Не покрыто | Покрытие |",
        "|---|---:|---:|---:|",
        (
            f"| src/generator_model.py | {cov['num_statements']} | "
            f"{cov['missing_lines']} | "
            f"{cov['percent_covered']:.2f}% |"
        ),
        (
            "Тесты проверяют схему, границы, все ручки, зерно, "
            "режимы и ошибки входных данных."
        ),
        "",
        "## Выводы",
        (
            f"1. Similar соответствует порогу на "
            f"{similar_fraction:.1%} признаков; "
            "вердикт приведён выше."
        ),
        (
            f"2. Random имеет PSI >= 0.2 на "
            f"{random_fraction:.1%} признаков; "
            "отклонения или пояснение приведены выше."
        ),
        (
            f"3. Побайтовая проверка: "
            f"similar={repeats['similar']}, "
            f"random={repeats['random']}."
        ),
        (
            "4. Данные бутстрэпа могут повторять реальные строки "
            "и не являются анонимизацией; перед внешним применением "
            "нужна оценка рисков."
        ),
        "",
    ]

    (out / "lab5_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"similar: доля признаков <= 0.1 = {similar_fraction:.3f}; "
        f"random PSI >= 0.2 = {random_fraction:.3f}"
    )

    print(
        f"Побайтовые повторы: {repeats}; "
        f"покрытие: {cov['percent_covered']:.2f}%; "
        f"отчёт: {out / 'lab5_report.md'}"
    )

    if similar_fraction < 0.8 or not all(repeats.values()):
        print("Обязательный критерий не выполнен — изучите отчёт.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
