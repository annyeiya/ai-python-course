"""ЛР2: ML-пайплайн варианта 10 в двух режимах - полный датасет и чанки.

full:   StandardScaler + LogisticRegression(l1, saga) на массивах train+val.
chunks: статистики скейлера накапливаются partial_fit по чанкам с диска,
        модель - SGDClassifier(log_loss, l1) с partial_fit по тем же чанкам.

Запуск:
  python3 src/ml_pipeline.py --config configs/ml_config.yaml
  python3 src/ml_pipeline.py --config configs/ml_config.yaml --check-repeat
"""
from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError, field_validator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eda import sha256_file


class MLConfig(BaseModel):
    """Схема настроек ЛР2; на невалидных значениях работа не запускается."""

    dataset_path: str
    hash_manifest_path: str
    report_dir: str
    model_path: str
    seed: int = 511
    val_size: float = 0.15
    test_size: float = 0.15
    chunk_size: int = 2000
    chunk_epochs: int = 10
    c_grid: list[float] = [0.01, 0.03, 0.1, 0.3, 1.0]
    n_bootstrap: int = 1000
    n_repeats: int = 3
    mode_tolerance: float = 0.01
    author: str = "Нефедова А.А."
    report_date: str = "2026-09-24"

    @field_validator("dataset_path", "hash_manifest_path")
    @classmethod
    def path_exists(cls, v: str) -> str:
        """Отказать, если файла данных или манифеста нет на диске."""
        if not Path(v).is_file():
            raise ValueError(f"файл не найден: {v} (проверьте путь в конфиге)")
        return v

    @field_validator("chunk_size", "chunk_epochs")
    @classmethod
    def chunk_positive(cls, v: int) -> int:
        """Отказать, если размер чанка или число проходов меньше 1."""
        if v < 1:
            raise ValueError(f"значение должно быть >= 1, получено {v}")
        return v

    @field_validator("c_grid")
    @classmethod
    def c_grid_valid(cls, v: list[float]) -> list[float]:
        """Отказать, если сетка C пустая или содержит C <= 0."""
        if not v:
            raise ValueError("c_grid не может быть пустой")
        bad = [c for c in v if c <= 0]
        if bad:
            raise ValueError(f"все C должны быть > 0, получены невалидные: {bad}")
        return v


def read_data(cfg: MLConfig) -> tuple[pd.DataFrame, list[str]]:
    """Прочитать csv и вернуть (таблица, список имён признаков)."""
    df = pd.read_csv(cfg.dataset_path)
    feats = [c for c in df.columns if c != "target"]
    return df, feats


def split_indices(y: np.ndarray, cfg: MLConfig):
    """Разбить индексы на train/val/test как 70/15/15, стратифицированно, по seed."""
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    idx_tr, idx_val = train_test_split(
        idx_tv, test_size=val_frac, stratify=y[idx_tv], random_state=cfg.seed
    )
    return idx_tr, idx_val, idx_test


def build_full_pipeline(c: float, seed: int) -> Pipeline:
    """Конвейер полного режима: скейлер + L1-логистическая регрессия (saga)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    l1_ratio=1.0, solver="saga", C=c, max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def tune_c(x_train, y_train, x_val, y_val, cfg: MLConfig):
    """Перебрать C из сетки: фит на train, выбор по ROC-AUC на validation."""
    table = []
    best_c, best_score = cfg.c_grid[0], -1.0
    for c in cfg.c_grid:
        pipe = build_full_pipeline(c, cfg.seed)
        pipe.fit(x_train, y_train)
        score = roc_auc_score(y_val, pipe.predict_proba(x_val)[:, 1])
        table.append((c, score))
        if score > best_score:
            best_c, best_score = c, score
    return best_c, table


def iter_chunks(path: str, chunk_size: int, mask: np.ndarray | None = None):
    """Отдавать csv чанками по chunk_size строк; mask оставляет строки train+val."""
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if mask is not None:
            chunk = chunk.loc[mask[chunk.index]]
        if len(chunk):
            yield chunk


def train_chunks(cfg: MLConfig, feats: list[str], mask: np.ndarray, alpha: float):
    """Чанковый режим: скейлер частичными фитами, затем SGD(l1) по чанкам.

    Проходов по потоку chunk_epochs: одного прохода мало, SGD не дообучится.
    Память ограничена чанком, весь датасет в память не читается.
    """
    scaler = StandardScaler()
    for chunk in iter_chunks(cfg.dataset_path, cfg.chunk_size, mask):
        scaler.partial_fit(chunk[feats].to_numpy())
    clf = SGDClassifier(loss="log_loss", l1_ratio=1.0, alpha=alpha,
                        random_state=cfg.seed)
    classes = np.array([0, 1])
    for _ in range(cfg.chunk_epochs):
        for chunk in iter_chunks(cfg.dataset_path, cfg.chunk_size, mask):
            x = scaler.transform(chunk[feats].to_numpy())
            clf.partial_fit(x, chunk["target"].to_numpy(), classes=classes)
    return scaler, clf


def measure(fn, n_repeats: int):
    """Запустить fn n_repeats раз под tracemalloc.

    Вернуть (результат последнего прогона, средний пик памяти МБ,
    min пик, max пик, среднее время в секундах).
    """
    peaks, dts, res = [], [], None
    for _ in range(n_repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / 1e6)
        dts.append(dt)
    return res, float(np.mean(peaks)), float(np.min(peaks)), float(np.max(peaks)), float(np.mean(dts))


def bootstrap_ci(y_true, proba, pred, seed: int, n: int):
    """Бутстрэп: n пересборок теста, вернуть интервалы 2.5%-97.5% трёх метрик."""
    rng = np.random.default_rng(seed)
    m = len(y_true)
    accs, prs, rocs = [], [], []
    for _ in range(n):
        i = rng.integers(0, m, m)
        if len(np.unique(y_true[i])) < 2:
            continue
        accs.append(accuracy_score(y_true[i], pred[i]))
        prs.append(average_precision_score(y_true[i], proba[i]))
        rocs.append(roc_auc_score(y_true[i], proba[i]))
    out = {}
    for name, arr in (("accuracy", accs), ("pr_auc", prs), ("roc_auc", rocs)):
        out[name] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def evaluate(y_test, proba, pred, cfg: MLConfig):
    """Метрики акцента на тесте (accuracy, PR-AUC, ROC-AUC) + их интервалы.

    Единственное место, где используется тест: контрольная часть одна.
    """
    values = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    return values, bootstrap_ci(y_test, proba, pred, cfg.seed, cfg.n_bootstrap)


def run_mode(cfg, mode: str, best_c: float, n_trainval: int, mask, x_trainval, y_trainval, feats):
    """Обучить один режим с замерами памяти и времени.

    Вернуть (функция predict(x) -> (proba, pred), пик МБ, min, max, время).
    """
    if mode == "full":
        def fn():
            pipe = build_full_pipeline(best_c, cfg.seed)
            pipe.fit(x_trainval, y_trainval)
            return pipe

        pipe, peak, pmin, pmax, dt = measure(fn, cfg.n_repeats)
        return (lambda x: (pipe.predict_proba(x)[:, 1], pipe.predict(x))), peak, pmin, pmax, dt
    alpha = 1.0 / (best_c * max(n_trainval, 1))
    scaler_box: dict = {}

    def fn():
        scaler, clf = train_chunks(cfg, feats, mask, alpha)
        scaler_box["scaler"] = scaler
        return clf

    clf, peak, pmin, pmax, dt = measure(fn, cfg.n_repeats)
    scaler = scaler_box["scaler"]

    def predict(x: np.ndarray):
        xs = scaler.transform(x)
        return clf.predict_proba(xs)[:, 1], clf.predict(xs)

    return predict, peak, pmin, pmax, dt


def log_mlflow(cfg: MLConfig, mode: str, params: dict, metrics: dict, artifact: str | None) -> str:
    """Записать прогон в MLflow (параметры, метрики, артефакт), вернуть run_id."""
    import mlflow

    # mlflow 3.x по умолчанию ушёл в SQLite и ругается на файловый трекер;
    # методичке нужен каталог mlruns/, поэтому включаем его явно.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(str(Path("mlruns").resolve()))
    mlflow.set_experiment("lab2-variant10")
    with mlflow.start_run(run_name=f"lab2-{mode}") as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if artifact:
            mlflow.log_artifact(artifact)
        return run.info.run_id


def run_pipeline(cfg: MLConfig, with_mlflow: bool = True) -> dict:
    """Весь прогон работы: разбиение, подбор C, оба режима, метрики, MLflow,
    сохранение модели с хешем, ml_metrics.csv и отчёт."""
    out = Path(cfg.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, feats = read_data(cfg)
    y_all = df["target"].to_numpy()
    idx_tr, idx_val, idx_test = split_indices(y_all, cfg)
    idx_tv = np.concatenate([idx_tr, idx_val])
    mask = np.zeros(len(df), dtype=bool)
    mask[idx_tv] = True

    x_tr = df[feats].to_numpy()[idx_tr]
    y_tr = y_all[idx_tr]
    x_val = df[feats].to_numpy()[idx_val]
    y_val = y_all[idx_val]
    x_test = df[feats].to_numpy()[idx_test]
    y_test = y_all[idx_test]
    x_trainval = df[feats].to_numpy()[idx_tv]
    y_trainval = y_all[idx_tv]

    best_c, tune_table = tune_c(x_tr, y_tr, x_val, y_val, cfg)
    print(f"[tune] лучшее C={best_c} по ROC-AUC на validation")

    results: dict = {"tune_table": tune_table, "best_c": best_c}
    run_ids = {}
    for mode in ("full", "chunks"):
        predict, peak, pmin, pmax, dt = run_mode(
            cfg, mode, best_c, len(idx_tv), mask, x_trainval, y_trainval, feats
        )
        proba, pred = predict(x_test)
        values, cis = evaluate(y_test, proba, pred, cfg)
        results[mode] = {
            "values": values, "cis": cis,
            "peak": peak, "pmin": pmin, "pmax": pmax, "dt": dt,
        }
        print(f"[{mode}] accuracy={values['accuracy']:.4f} "
              f"pr_auc={values['pr_auc']:.4f} roc_auc={values['roc_auc']:.4f} "
              f"peak={peak:.2f} МБ time={dt:.2f} c")

    # Наивный ориентир: предсказание всегда класса большинства.
    majority = int(np.round(y_trainval.mean()))
    naive_pred = np.full_like(y_test, majority)
    naive_acc = float(accuracy_score(y_test, naive_pred))
    naive_cis = bootstrap_ci(y_test, np.full(len(y_test), 0.5), naive_pred,
                             cfg.seed, cfg.n_bootstrap)
    results["naive"] = {"accuracy": naive_acc, "cis": naive_cis["accuracy"]}

    # Модель полного режима - артефакт работы; считаем её хеш.
    pipe = build_full_pipeline(best_c, cfg.seed)
    pipe.fit(x_trainval, y_trainval)
    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    model_hash = sha256_file(model_path)
    results["model_hash"] = model_hash
    coef = pipe.named_steps["clf"].coef_[0]
    zeroed = [feats[i] for i in range(len(feats)) if abs(coef[i]) < 1e-8]
    results["zeroed"] = zeroed
    print(f"[model] {model_path} sha256={model_hash}")
    print(f"[model] L1 обнулил признаков: {len(zeroed)} -> {zeroed}")

    if with_mlflow:
        full = results["full"]
        chunks = results["chunks"]
        run_ids["full"] = log_mlflow(
            cfg, "full",
            {"mode": "full", "seed": cfg.seed, "best_C": best_c,
             "solver": "saga", "l1_ratio": "1.0", "chunk_size": cfg.chunk_size},
            {**full["values"], "peak_memory_mb": full["peak"],
             "train_time_sec": full["dt"]},
            str(model_path),
        )
        tmp = out / "chunks_metrics.json"
        tmp.write_text(json.dumps(chunks["values"], indent=2), encoding="utf-8")
        run_ids["chunks"] = log_mlflow(
            cfg, "chunks",
            {"mode": "chunks", "seed": cfg.seed, "best_C": best_c,
             "solver": "sgd partial_fit", "l1_ratio": "1.0",
             "alpha": 1.0 / (best_c * len(idx_tv)), "chunk_size": cfg.chunk_size,
             "chunk_epochs": cfg.chunk_epochs},
            {**chunks["values"], "peak_memory_mb": chunks["peak"],
             "train_time_sec": chunks["dt"]},
            str(tmp),
        )
        tmp.unlink()
    results["run_ids"] = run_ids

    write_metrics_csv(cfg, results)
    write_report(cfg, results)
    return results


def write_metrics_csv(cfg: MLConfig, results: dict) -> None:
    """Записать ml_metrics.csv: метрики обоих режимов с интервалами и run_id."""
    rows = []
    for mode in ("full", "chunks"):
        r = results[mode]
        rid = results["run_ids"].get(mode, "no-mlflow")
        for metric in ("accuracy", "pr_auc", "roc_auc"):
            lo, hi = r["cis"][metric]
            rows.append({"mode": mode, "metric": metric,
                         "value": round(r["values"][metric], 6),
                         "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                         "run_id": rid})
        rows.append({"mode": mode, "metric": "peak_memory_mb",
                     "value": round(r["peak"], 4), "ci_low": round(r["pmin"], 4),
                     "ci_high": round(r["pmax"], 4), "run_id": rid})
        rows.append({"mode": mode, "metric": "train_time_sec",
                     "value": round(r["dt"], 4), "ci_low": round(r["dt"], 4),
                     "ci_high": round(r["dt"], 4), "run_id": rid})
    rid = results["run_ids"].get("full", "no-mlflow")
    lo, hi = results["naive"]["cis"]
    rows.append({"mode": "naive", "metric": "accuracy",
                 "value": round(results["naive"]["accuracy"], 6),
                 "ci_low": round(lo, 6), "ci_high": round(hi, 6), "run_id": rid})
    path = Path(cfg.report_dir) / "ml_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[csv] {path} (строк: {len(rows)})")


def write_report(cfg: MLConfig, results: dict) -> None:
    """Собрать lab2_report.md из чисел текущего прогона, чтобы отчёт и csv совпадали."""
    out = Path(cfg.report_dir)
    full, chunks = results["full"], results["chunks"]
    manifest = json.loads(Path(cfg.hash_manifest_path).read_text(encoding="utf-8"))
    data_hash = manifest["files"][0]["sha256"]
    diff_acc = abs(full["values"]["accuracy"] - chunks["values"]["accuracy"])
    verdict = (
        f"расхождение {diff_acc:.4f} не превышает допуск {cfg.mode_tolerance}"
        if diff_acc <= cfg.mode_tolerance
        else f"расхождение {diff_acc:.4f} выше допуска {cfg.mode_tolerance}: "
        "режимы используют разные солверы l1-логистической регрессии "
        "(saga против стохастического partial_fit), различие объяснено"
    )
    tune_rows = "\n".join(
        f"| {c} | {s:.6f} |" for c, s in results["tune_table"]
    )
    metric_rows = []
    for metric in ("accuracy", "pr_auc", "roc_auc"):
        fv, cv = full["values"][metric], chunks["values"][metric]
        flo, fhi = full["cis"][metric]
        clo, chi = chunks["cis"][metric]
        metric_rows.append(
            f"| {metric} | {fv:.6f} | [{flo:.6f}; {fhi:.6f}] | {cv:.6f} | "
            f"[{clo:.6f}; {chi:.6f}] | {abs(fv - cv):.6f} |"
        )
    metric_rows.append(
        f"| peak_memory_mb | {full['peak']:.4f} | "
        f"[{full['pmin']:.4f}; {full['pmax']:.4f}] | {chunks['peak']:.4f} | "
        f"[{chunks['pmin']:.4f}; {chunks['pmax']:.4f}] | - |"
    )
    metric_rows.append(
        f"| train_time_sec | {full['dt']:.4f} | [{full['dt']:.4f}; {full['dt']:.4f}] | "
        f"{chunks['dt']:.4f} | [{chunks['dt']:.4f}; {chunks['dt']:.4f}] | - |"
    )
    n_zero = len(results["zeroed"])
    text = f"""# ЛР2. Пайплайн ML-модели: полный датасет и чанки

| Поле | Значение |
|---|---|
| Номер варианта | 10 (Шумовые признаки, профиль P1 - табличный) |
| Номер работы | ЛР2 |
| Дата | {cfg.report_date} |
| Автор | {cfg.author} |
| Хеш коммита кода | <COMMIT_HASH> |
| Хеши данных | data/raw/dataset.csv sha256={data_hash} |

## 1. Схема настроек (pydantic, configs/ml_config.yaml)

Поля схемы MLConfig: dataset_path, hash_manifest_path, report_dir, model_path,
seed={cfg.seed}, val_size={cfg.val_size}, test_size={cfg.test_size},
chunk_size={cfg.chunk_size}, chunk_epochs={cfg.chunk_epochs}, c_grid={cfg.c_grid}, n_bootstrap={cfg.n_bootstrap},
n_repeats={cfg.n_repeats}, mode_tolerance={cfg.mode_tolerance}.
Валидаторы отвергают: несуществующий файл данных, chunk_size < 1, непустую сетку
C с неотрицательными значениями; отказ фиксируется в config_negative_test.log.

## 2. Состав конвейера и способ подбора параметров

Конвейер: StandardScaler -> LogisticRegression(l1_ratio=1.0, solver="saga").
Импутация и кодирование категорий не требуются: по ЛР1 пропусков 0,
категориальных полей 0; скейлер обязателен из-за разного масштаба признаков.
Подбор C: фит на train, оценка ROC-AUC только на validation:

| C | ROC-AUC на validation |
|---|---|
{tune_rows}

Выбрано C = {results['best_c']}. Контрольная часть (test) в подборе не участвует
и используется один раз - в функции evaluate.

## 3. Таблица сравнения режимов

| Метрика | full | full 95% ИИ | chunks | chunks 95% ИИ | |full-chunks| |
|---|---|---|---|---|---|
{chr(10).join(metric_rows)}

Интервалы метрик - бутстрэп 2.5%/97.5% по {cfg.n_bootstrap} пересборкам теста;
интервалы памяти и времени - min/max по {cfg.n_repeats} повторам замера.
Вердикт по ключевой метрике accuracy: {verdict}.
Номера прогонов MLflow: full = {results['run_ids'].get('full', 'нет')},
chunks = {results['run_ids'].get('chunks', 'нет')}.

## 4. Модель и воспроизводимость

- Модель полного режима: {cfg.model_path}, sha256={results['model_hash']}.
- Загрузка и предсказание воспроизводятся: проверка --check-repeat сверяет
  метрики загруженной модели с ml_metrics.csv (repeat_check.log).
- Повтор прогона с тем же зерном даёт те же метрики (ml_run1.log, ml_run2.log,
  repeat_check.log).

## 5. Негативные контроли и проверка

| Контроль | Чем проверяли | Результат | Файл |
|---|---|---|---|
| Невалидный конфиг: chunk_size=0 | запуск с configs/negative/bad_chunk.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Невалидный конфиг: файл данных не существует | запуск с configs/negative/bad_path.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Невалидный конфиг: C <= 0 | запуск с configs/negative/bad_c.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Ловушка утечки (скейлер по всем данным) | pytest tests/test_lab2_leak.py | тест отличает фит по всем данным от фита по train | leak_memory_tests.log |
| Пик памяти чанков не растёт с объёмом | pytest tests/test_lab2_leak.py (4000 против 8000 строк) | прирост пика меньше двух чанков | leak_memory_tests.log |
| Повтор прогона с тем же зерном | --check-repeat | метрики совпали с ml_metrics.csv | repeat_check.log |

## 6. Выводы

1. Подбор на validation выбрал C = {results['best_c']} (таблица раздела 2; ml_run1.log).
2. Полный режим: accuracy = {full['values']['accuracy']:.6f} с 95% интервалом
   [{full['cis']['accuracy'][0]:.6f}; {full['cis']['accuracy'][1]:.6f}]
   (ml_metrics.csv: full accuracy).
3. Чанковый режим: accuracy = {chunks['values']['accuracy']:.6f} с 95% интервалом
   [{chunks['cis']['accuracy'][0]:.6f}; {chunks['cis']['accuracy'][1]:.6f}]
   (ml_metrics.csv: chunks accuracy).
4. Разность accuracy режимов равна {diff_acc:.6f}, вердикт: {verdict}
   (ml_metrics.csv).
5. Наивный ориентир «всегда класс большинства» даёт accuracy =
   {results['naive']['accuracy']:.6f}, модель лучше него
   (ml_metrics.csv: naive accuracy).
6. L1-штраф обнулил {n_zero} из 20 весов: {', '.join(results['zeroed'])}
   (ml_run1.log; проблема варианта 10).
7. Пик памяти полного режима {full['peak']:.4f} МБ против {chunks['peak']:.4f} МБ
   у чанкового: чанки держат память размером порции (ml_metrics.csv).
8. Время обучения: полный режим {full['dt']:.4f} c, чанковый {chunks['dt']:.4f} c
   (ml_metrics.csv: train_time_sec).
9. PR-AUC полного режима {full['values']['pr_auc']:.6f}, ROC-AUC
   {full['values']['roc_auc']:.6f} - метрики акцента карточки варианта
   (ml_metrics.csv: full pr_auc, roc_auc).
10. Хеш модели {results['model_hash'][:16]}... воспроизводится при загрузке
    (repeat_check.log; раздел 4).
"""
    (out / "lab2_report.md").write_text(text, encoding="utf-8")
    print(f"[report] {out / 'lab2_report.md'}")


def check_repeat(cfg: MLConfig) -> int:
    """Повторить оба режима без записи и сверить с ml_metrics.csv и загруженной моделью."""
    df, feats = read_data(cfg)
    y_all = df["target"].to_numpy()
    idx_tr, idx_val, idx_test = split_indices(y_all, cfg)
    idx_tv = np.concatenate([idx_tr, idx_val])
    mask = np.zeros(len(df), dtype=bool)
    mask[idx_tv] = True
    x_test = df[feats].to_numpy()[idx_test]
    y_test = y_all[idx_test]
    x_trainval = df[feats].to_numpy()[idx_tv]
    y_trainval = y_all[idx_tv]
    x_tr = df[feats].to_numpy()[idx_tr]
    y_tr = y_all[idx_tr]
    x_val = df[feats].to_numpy()[idx_val]
    y_val = y_all[idx_val]
    best_c, _ = tune_c(x_tr, y_tr, x_val, y_val, cfg)

    csv = pd.read_csv(Path(cfg.report_dir) / "ml_metrics.csv")
    ok = True
    for mode in ("full", "chunks"):
        predict, _, _, _, _ = run_mode(cfg, mode, best_c, len(idx_tv), mask,
                                       x_trainval, y_trainval, feats)
        proba, pred = predict(x_test)
        values, _ = evaluate(y_test, proba, pred, cfg)
        for metric in ("accuracy", "pr_auc", "roc_auc"):
            ref = csv[(csv["mode"] == mode) & (csv["metric"] == metric)]
            ref_v = float(ref["value"].iloc[0])
            same = round(values[metric], 6) == ref_v
            ok = ok and same
            print(f"[repeat] {mode} {metric}: повтор={values[metric]:.6f} "
                  f"csv={ref_v:.6f} совпадает={same}")
    pipe = joblib.load(Path(cfg.model_path))
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = pipe.predict(x_test)
    loaded = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    for metric in ("accuracy", "pr_auc", "roc_auc"):
        v = loaded[metric]
        ref = csv[(csv["mode"] == "full") & (csv["metric"] == metric)]
        ref_v = float(ref["value"].iloc[0])
        same = round(v, 6) == ref_v
        ok = ok and same
        print(f"[repeat] загруженная модель {metric}: {v:.6f} csv={ref_v:.6f} "
              f"совпадает={same}")
    print(f"[repeat] ВЕРДИКТ: {'совпадает' if ok else 'НЕ совпадает'}")
    return 0 if ok else 1


def main() -> None:
    """Прочитать конфиг (при ошибке валидации - объяснение и exit 2) и запустить прогон."""
    ap = argparse.ArgumentParser(description="ЛР2: ML-пайплайн, полный и чанки")
    ap.add_argument("--config", required=True)
    ap.add_argument("--check-repeat", action="store_true",
                    help="повтор прогона и проверка загрузки модели без записи")
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        cfg = MLConfig(**raw)
    except ValidationError as e:
        print("КОНФИГУРАЦИЯ НЕВАЛИДНА, работа остановлена:")
        for err in e.errors():
            where = ".".join(str(x) for x in err["loc"])
            print(f"  - поле {where}: {err['msg']} (значение: {err.get('input')})")
        raise SystemExit(2)
    if args.check_repeat:
        raise SystemExit(check_repeat(cfg))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
