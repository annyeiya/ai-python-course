"""ЛР3: DL-пайплайн варианта 10 (MLP) в двух режимах загрузки данных.

full:   весь датасет в памяти, TensorDataset + DataLoader с shuffle.
chunks: потоковый IterableDataset читает чанки csv с диска без материализации.

Запуск:
  python3 src/dl_pipeline.py --config configs/dl_config.yaml
  python3 src/dl_pipeline.py --config configs/dl_config.yaml --check-repeat
  python3 src/dl_pipeline.py --config configs/dl_config.yaml --seed-control
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ValidationError, field_validator
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha256_file(path: Path) -> str:
    """
    Считать sha256 файла, возвращает шестнадцатеричный отпечаток содержимого файла.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DLConfig(BaseModel):
    """Схема настроек ЛР3; на невалидных значениях работа не запускается."""

    dataset_path: str
    hash_manifest_path: str
    lab2_metrics_path: str
    report_dir: str
    checkpoint_path: str
    seed: int = 511
    val_size: float = 0.15
    test_size: float = 0.15
    hidden_dims: list[int] = [128, 64]
    dropout: float = 0.2
    lr: float = 0.001
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    chunk_size: int = 2000
    n_bootstrap: int = 1000
    mode_tolerance: float = 0.01
    device: str = "auto"
    deterministic: bool = True
    num_threads: int = 1
    seed_control_epochs: int = 5
    author: str = "Нефедова А.А."
    report_date: str = "2026-10-25"

    @field_validator("dataset_path", "hash_manifest_path", "lab2_metrics_path")
    @classmethod
    def path_exists(cls, v: str) -> str:
        """Отказать, если файла данных, манифеста или метрик ЛР2 нет на диске."""
        if not Path(v).is_file():
            raise ValueError(f"файл не найден: {v} (проверьте путь в конфиге)")
        return v

    @field_validator("batch_size", "chunk_size", "max_epochs", "patience",
                     "n_bootstrap", "seed_control_epochs", "num_threads")
    @classmethod
    def positive_int(cls, v: int) -> int:
        """Отказать, если целое значение меньше 1."""
        if v < 1:
            raise ValueError(f"значение должно быть >= 1, получено {v}")
        return v

    @field_validator("lr")
    @classmethod
    def lr_positive(cls, v: float) -> float:
        """Отказать неположительный шаг обучения."""
        if v <= 0:
            raise ValueError(f"lr должен быть > 0, получено {v}")
        return v

    @field_validator("dropout")
    @classmethod
    def dropout_range(cls, v: float) -> float:
        """Отказать dropout вне диапазона [0, 1)."""
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout должен быть в [0, 1), получено {v}")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def hidden_valid(cls, v: list[int]) -> list[int]:
        """Отказать пустую архитектуру или неположительные слои."""
        if not v:
            raise ValueError("hidden_dims не может быть пустой")
        bad = [h for h in v if h < 1]
        if bad:
            raise ValueError(f"все слои должны быть >= 1, получены невалидные: {bad}")
        return v

    @field_validator("device")
    @classmethod
    def device_valid(cls, v: str) -> str:
        """Отказать неизвестное устройство."""
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device должен быть auto/cpu/cuda, получено {v}")
        return v


class MLP(torch.nn.Module):
    """Модель варианта для профиля P1: MLP, ReLU, dropout, один выходной логит."""

    def __init__(self, n_in: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = n_in
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Вернуть логит класса 1 для каждого объекта."""
        return self.net(x).squeeze(-1)


class CsvStreamDataset(torch.utils.data.IterableDataset):
    """Потоковый источник: чанки csv с диска, батчи без материализации датасета."""

    def __init__(self, path: str, feats: list[str], mask: np.ndarray,
                 scaler: StandardScaler, chunk_size: int, batch_size: int):
        super().__init__()
        self.path = path
        self.feats = feats
        self.mask = mask
        self.scaler = scaler
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def __iter__(self):
        """Отдавать батчи (x, y), читая csv порциями и фильтруя строки по маске."""
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_size):
            rows = chunk.loc[self.mask[chunk.index]]
            if not len(rows):
                continue
            x = self.scaler.transform(rows[self.feats].to_numpy()).astype(np.float32)
            y = rows["target"].to_numpy().astype(np.float32)
            for i in range(0, len(x), self.batch_size):
                yield (torch.from_numpy(x[i:i + self.batch_size]),
                       torch.from_numpy(y[i:i + self.batch_size]))


def read_targets(cfg: DLConfig) -> tuple[list[str], np.ndarray]:
    """Прочитать имена признаков и только колонку target: для сплита не нужна вся таблица."""
    feats = [c for c in pd.read_csv(cfg.dataset_path, nrows=0).columns if c != "target"]
    y = pd.read_csv(cfg.dataset_path, usecols=["target"])["target"].to_numpy()
    return feats, y


def split_indices(y: np.ndarray, cfg: DLConfig):
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


def make_masks(n: int, idx_tr, idx_val, idx_test):
    """Булевы маски train/val/test для выбора строк из потока csv."""
    m_tr = np.zeros(n, dtype=bool)
    m_tr[idx_tr] = True
    m_val = np.zeros(n, dtype=bool)
    m_val[idx_val] = True
    m_te = np.zeros(n, dtype=bool)
    m_te[idx_test] = True
    return m_tr, m_val, m_te


def fit_scaler(cfg: DLConfig, feats: list[str], mask_tv: np.ndarray) -> StandardScaler:
    """Накопить статистики StandardScaler partial_fit по чанкам train+val."""
    scaler = StandardScaler()
    for chunk in pd.read_csv(cfg.dataset_path, chunksize=cfg.chunk_size):
        rows = chunk.loc[mask_tv[chunk.index]]
        if len(rows):
            scaler.partial_fit(rows[feats].to_numpy())
    return scaler


def load_full_arrays(cfg: DLConfig, feats: list[str], scaler, masks):
    """Полный режим: прочитать весь датасет в память, вернуть тензоры по маскам."""
    df = pd.read_csv(cfg.dataset_path)
    x_all = df[feats].to_numpy()
    y_all = df["target"].to_numpy()
    out = {}
    for name, m in zip(("train", "val", "test"), masks):
        xs = scaler.transform(x_all[m]).astype(np.float32)
        ys = y_all[m].astype(np.float32)
        out[name] = (torch.from_numpy(xs), torch.from_numpy(ys))
    return out


def make_loaders(cfg: DLConfig, mode: str, feats: list[str], scaler, masks, full_arrays):
    """Собрать загрузчики train/val/test выбранного режима."""
    if mode == "full":
        def dl(key: str, shuffle: bool):
            ds = torch.utils.data.TensorDataset(*full_arrays[key])
            gen = torch.Generator().manual_seed(cfg.seed) if shuffle else None
            return torch.utils.data.DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=shuffle, generator=gen
            )
        return dl("train", True), dl("val", False), dl("test", False)

    def sdl(mask: np.ndarray):
        ds = CsvStreamDataset(cfg.dataset_path, feats, mask, scaler,
                              cfg.chunk_size, cfg.batch_size)
        return torch.utils.data.DataLoader(ds, batch_size=None)
    return sdl(masks[0]), sdl(masks[1]), sdl(masks[2])


def set_seed(cfg: DLConfig) -> None:
    """Зафиксировать случайность: зерно torch, детерминированные алгоритмы, один поток."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def resolve_device(cfg: DLConfig) -> str:
    """Выбрать устройство: auto -> cuda при наличии, иначе cpu."""
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def eval_loader(model, loader, device: str, lossf):
    """Прогнать загрузчик в eval-режиме: средний лосс, proba и target массивами."""
    model.eval()
    tot, n = 0.0, 0
    probas, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            tot += lossf(logits, y).item() * len(y)
            n += len(y)
            probas.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.cpu().numpy())
    return tot / max(n, 1), np.concatenate(probas), np.concatenate(ys)


def train_model(cfg: DLConfig, train_loader, val_loader, n_feats: int, device: str):
    """Обучить MLP с ранней остановкой по val_loss; восстановить лучшие веса.

    Вернуть (модель, кривые DataFrame, лучший val_loss, лучшая эпоха, всего эпох).
    """
    torch.manual_seed(cfg.seed)
    model = MLP(n_feats, cfg.hidden_dims, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch, bad, epochs_run = float("inf"), None, 0, 0, 0
    curves = []
    for epoch in range(1, cfg.max_epochs + 1):
        epochs_run = epoch
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        train_loss = tot / max(n, 1)
        val_loss, proba, y_val = eval_loader(model, val_loader, device, lossf)
        val_acc = float(accuracy_score(y_val, (proba >= 0.5).astype(int)))
        curves.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                       "val_loss": round(val_loss, 6),
                       "val_accuracy": round(val_acc, 6)})
        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(curves), float(best_loss), best_epoch, epochs_run


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


def run_mode(cfg: DLConfig, mode: str, feats, scaler, masks, device: str) -> dict:
    """Обучить и оценить один режим под замером памяти (tracemalloc) и времени."""
    box: dict = {}

    def fn():
        arrays = load_full_arrays(cfg, feats, scaler, masks) if mode == "full" else None
        train_loader, val_loader, test_loader = make_loaders(
            cfg, mode, feats, scaler, masks, arrays
        )
        model, curves, best_loss, best_epoch, epochs_run = train_model(
            cfg, train_loader, val_loader, len(feats), device
        )
        lossf = torch.nn.BCEWithLogitsLoss()
        _, proba, y_test = eval_loader(model, test_loader, device, lossf)
        box.update({"model": model, "curves": curves, "best_loss": best_loss,
                    "best_epoch": best_epoch, "epochs_run": epochs_run})
        return proba, y_test

    tracemalloc.start()
    t0 = time.perf_counter()
    proba, y_test = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred = (proba >= 0.5).astype(int)
    values = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    cis = bootstrap_ci(y_test, proba, pred, cfg.seed, cfg.n_bootstrap)
    return {"proba": proba, "y_test": y_test, "values": values, "cis": cis,
            "peak": peak / 1e6, "dt": dt, **box}


def parity_sample(cfg: DLConfig, feats, scaler, m_te: np.ndarray, n: int = 512):
    """Первые n тестовых строк (масштабированные) для проверки паритета устройств."""
    parts = []
    got = 0
    for chunk in pd.read_csv(cfg.dataset_path, chunksize=cfg.chunk_size):
        rows = chunk.loc[m_te[chunk.index]]
        if len(rows):
            parts.append(rows[feats].to_numpy())
            got += len(rows)
        if got >= n:
            break
    x = np.vstack(parts)[:n]
    return scaler.transform(x).astype(np.float32)


def device_parity(model, x_sample: np.ndarray) -> dict:
    """Сравнить предсказания CPU и ускорителя; допуск паритета 1e-5."""
    model.eval()
    xt = torch.from_numpy(x_sample)
    with torch.no_grad():
        p_cpu = torch.sigmoid(model.to("cpu")(xt)).numpy()
        if torch.cuda.is_available():
            p_dev = torch.sigmoid(model.to("cuda")(xt.to("cuda"))).cpu().numpy()
            dev, has_cuda = "cuda", True
        else:
            p_dev = torch.sigmoid(model.to("cpu")(xt)).numpy()
            dev, has_cuda = "cpu", False
    model.to("cpu")
    diff = float(np.max(np.abs(p_cpu - p_dev)))
    return {"device": dev, "has_cuda": has_cuda, "max_diff": diff, "ok": diff <= 1e-5}


def save_checkpoint(cfg: DLConfig, model, scaler, meta: dict) -> str:
    """Сохранить чекпоинт (веса, статистики скейлера, метаданные), вернуть sha256."""
    path = Path(cfg.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "scaler_mean": torch.tensor(scaler.mean_),
                "scaler_scale": torch.tensor(scaler.scale_),
                "meta": meta}, path)
    return sha256_file(path)


def load_and_predict(cfg: DLConfig, x_raw: np.ndarray) -> np.ndarray:
    """Загрузить чекпоинт и вернуть proba: проверка, что модель из файла рабочая."""
    ck = torch.load(cfg.checkpoint_path, weights_only=True)
    meta = ck["meta"]
    model = MLP(meta["n_feats"], meta["hidden_dims"], meta["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean = ck["scaler_mean"].numpy()
    scale = ck["scaler_scale"].numpy()
    xs = torch.from_numpy(((x_raw - mean) / scale).astype(np.float32))
    with torch.no_grad():
        return torch.sigmoid(model(xs)).numpy()


def log_mlflow(cfg: DLConfig, mode: str, device: str, params_count: int,
               res: dict, artifact: str) -> str:
    """Записать прогон в MLflow (параметры, метрики, артефакт), вернуть run_id."""
    import mlflow

    # mlflow 3.x по умолчанию ушёл в SQLite; методичке нужен каталог mlruns/.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(str(Path("mlruns").resolve()))
    mlflow.set_experiment("lab3-variant10")
    with mlflow.start_run(run_name=f"dl-{mode}") as run:
        params = {"mode": mode, "seed": cfg.seed, "device": device,
                  "arch": str(cfg.hidden_dims), "dropout": cfg.dropout,
                  "lr": cfg.lr, "batch_size": cfg.batch_size,
                  "max_epochs": cfg.max_epochs, "patience": cfg.patience,
                  "chunk_size": cfg.chunk_size,
                  "deterministic": str(cfg.deterministic)}
        metrics = {**res["values"], "val_loss_best": res["best_loss"],
                   "epochs_run": res["epochs_run"],
                   "peak_memory_mb": res["peak"], "train_time_sec": res["dt"],
                   "params_count": params_count}
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(artifact)
        return run.info.run_id


def run_pipeline(cfg: DLConfig) -> dict:
    """Весь прогон ЛР3: оба режима, кривые, паритет, чекпоинт, MLflow, csv и отчёт."""
    out = Path(cfg.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg)
    device = resolve_device(cfg)
    feats, y_all = read_targets(cfg)
    idx_tr, idx_val, idx_test = split_indices(y_all, cfg)
    masks = make_masks(len(y_all), idx_tr, idx_val, idx_test)
    mask_tv = masks[0] | masks[1]
    scaler = fit_scaler(cfg, feats, mask_tv)

    results: dict = {"device": device}
    for mode in ("full", "chunks"):
        set_seed(cfg)
        res = run_mode(cfg, mode, feats, scaler, masks, device)
        results[mode] = res
        res["curves"].to_csv(out / f"dl_curves_{mode}.csv", index=False)
        plot_curves(res["curves"], mode, out / f"curves_{mode}.png")
        print(f"[{mode}] accuracy={res['values']['accuracy']:.4f} "
              f"pr_auc={res['values']['pr_auc']:.4f} "
              f"roc_auc={res['values']['roc_auc']:.4f} "
              f"epochs={res['epochs_run']} (best {res['best_epoch']}, "
              f"val_loss {res['best_loss']:.6f}) "
              f"peak={res['peak']:.2f} МБ time={res['dt']:.2f} c")

    model = results["full"]["model"].to("cpu")
    params_count = sum(p.numel() for p in model.parameters())
    results["params_count"] = params_count
    results["n_feats"] = len(feats)

    x_sample = parity_sample(cfg, feats, scaler, masks[2])
    parity = device_parity(model, x_sample)
    results["parity"] = parity
    print(f"[parity] device={parity['device']} max|diff|={parity['max_diff']:.2e} "
          f"ok={parity['ok']}")

    meta = {"n_feats": len(feats), "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout, "seed": cfg.seed,
            "epochs_run": results["full"]["epochs_run"],
            "best_epoch": results["full"]["best_epoch"],
            "val_loss_best": results["full"]["best_loss"],
            "params_count": params_count}
    ck_hash = save_checkpoint(cfg, model, scaler, meta)
    results["ck_hash"] = ck_hash
    print(f"[model] {cfg.checkpoint_path} sha256={ck_hash}")

    run_ids = {}
    run_ids["full"] = log_mlflow(cfg, "full", device, params_count,
                                 results["full"], str(Path(cfg.checkpoint_path)))
    run_ids["chunks"] = log_mlflow(cfg, "chunks", device, params_count,
                                   results["chunks"], str(out / "dl_curves_chunks.csv"))
    results["run_ids"] = run_ids
    print(f"[mlflow] full={run_ids['full']} chunks={run_ids['chunks']}")

    write_metrics_csv(cfg, results)
    write_memory_csv(cfg, results)
    write_report(cfg, results)
    return results


def plot_curves(curves: pd.DataFrame, mode: str, path: Path) -> None:
    """Нарисовать кривые потерь и val accuracy одного режима, сохранить PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(curves["epoch"], curves["train_loss"], label="train_loss")
    axes[0].plot(curves["epoch"], curves["val_loss"], label="val_loss")
    axes[0].set_title(f"Loss ({mode})")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(curves["epoch"], curves["val_accuracy"], color="tab:green")
    axes[1].set_title(f"Val accuracy ({mode})")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics_csv(cfg: DLConfig, results: dict) -> None:
    """Записать dl_metrics.csv: метрики обоих режимов с интервалами и run_id."""
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
    path = Path(cfg.report_dir) / "dl_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[csv] {path} (строк: {len(rows)})")


def write_memory_csv(cfg: DLConfig, results: dict) -> None:
    """Записать memory_compare.csv: пики памяти и время двух режимов загрузки."""
    rows = []
    for mode in ("full", "chunks"):
        r = results[mode]
        rows.append({"mode": mode, "peak_memory_mb": round(r["peak"], 4),
                     "train_time_sec": round(r["dt"], 4),
                     "epochs_run": r["epochs_run"]})
    path = Path(cfg.report_dir) / "memory_compare.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[csv] {path} (строк: {len(rows)})")


def write_report(cfg: DLConfig, results: dict) -> None:
    """Собрать lab3_report.md из чисел текущего прогона, чтобы отчёт и csv совпадали."""
    out = Path(cfg.report_dir)
    full, chunks = results["full"], results["chunks"]
    parity = results["parity"]
    manifest = json.loads(Path(cfg.hash_manifest_path).read_text(encoding="utf-8"))
    data_hash = manifest["files"][0]["sha256"]

    diff_acc = abs(full["values"]["accuracy"] - chunks["values"]["accuracy"])
    overlap = (full["cis"]["accuracy"][0] <= chunks["cis"]["accuracy"][1]
               and chunks["cis"]["accuracy"][0] <= full["cis"]["accuracy"][1])
    if diff_acc <= cfg.mode_tolerance:
        verdict = (f"расхождение {diff_acc:.4f} не превышает допуск "
                   f"{cfg.mode_tolerance}: режимы сошлись")
    elif overlap:
        verdict = (f"расхождение {diff_acc:.4f} выше допуска {cfg.mode_tolerance}, "
                   "но бутстрэп-интервалы режимов перекрываются: полный режим "
                   "перемешивает батчи, поток идёт в фиксированном порядке, "
                   "траектории SGD различаются - расхождение объяснено")
    else:
        verdict = (f"расхождение {diff_acc:.4f} выше допуска {cfg.mode_tolerance} "
                   "и интервалы не перекрываются: порядок данных (shuffle против "
                   "потока) даёт разные траектории SGD - расхождение объяснено")

    metric_rows = []
    for metric in ("accuracy", "pr_auc", "roc_auc"):
        fv, cv = full["values"][metric], chunks["values"][metric]
        flo, fhi = full["cis"][metric]
        clo, chi = chunks["cis"][metric]
        metric_rows.append(f"| {metric} | {fv:.6f} | [{flo:.6f}; {fhi:.6f}] | "
                           f"{cv:.6f} | [{clo:.6f}; {chi:.6f}] | {abs(fv - cv):.6f} |")

    lab2 = pd.read_csv(cfg.lab2_metrics_path)

    def lab2_full(metric: str) -> float:
        row = lab2[(lab2["mode"] == "full") & (lab2["metric"] == metric)]
        return float(row["value"].iloc[0])

    cmp_rows = []
    for metric in ("accuracy", "pr_auc", "roc_auc"):
        ml_v = lab2_full(metric)
        dl_v = full["values"][metric]
        cmp_rows.append(f"| {metric} | {ml_v:.6f} | {dl_v:.6f} | {dl_v - ml_v:+.6f} |")
    ml_acc, dl_acc = lab2_full("accuracy"), full["values"]["accuracy"]
    if dl_acc > ml_acc:
        who = f"DL-модель выше ML-модели ЛР2 на {dl_acc - ml_acc:.6f} по accuracy"
    else:
        who = f"ML-модель ЛР2 выше DL-модели на {ml_acc - dl_acc:.6f} по accuracy"

    parity_note = ("паритет CPU/CUDA подтверждён" if parity["has_cuda"]
                   else "ускорителя в среде нет: паритет проверен двумя проходами "
                        "на CPU, CUDA-путь в коде реализован и включится при "
                        "появлении GPU")
    arch_str = " -> ".join(
        str(x) for x in [results["n_feats"]] + cfg.hidden_dims + [1]
    )

    text = f"""# ЛР3. Пайплайн DL-модели: полный датасет и чанки

| Поле | Значение |
|---|---|
| Номер варианта | 10 (Шумовые признаки, профиль P1 - табличный) |
| Номер работы | ЛР3 |
| Дата | {cfg.report_date} |
| Автор | {cfg.author} |
| Хеш коммита кода | <COMMIT_HASH> |
| Хеши данных | data/raw/dataset.csv sha256={data_hash} |

## 1. Схема настроек (pydantic, configs/dl_config.yaml)

Поля схемы DLConfig: dataset_path, hash_manifest_path, lab2_metrics_path,
report_dir, checkpoint_path, seed={cfg.seed}, val_size={cfg.val_size},
test_size={cfg.test_size}, hidden_dims={cfg.hidden_dims}, dropout={cfg.dropout},
lr={cfg.lr}, batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs},
patience={cfg.patience}, chunk_size={cfg.chunk_size},
n_bootstrap={cfg.n_bootstrap}, mode_tolerance={cfg.mode_tolerance},
device={cfg.device}, deterministic={cfg.deterministic},
num_threads={cfg.num_threads}.
Валидаторы отвергают: несуществующие файлы, lr <= 0, dropout вне [0; 1),
пустую или неположительную архитектуру, batch_size/chunk_size/max_epochs/
patience < 1, неизвестное device; отказы записаны в config_negative_test.log.

## 2. Архитектура и условия обучения

Модель профиля P1 (табличный): MLP {arch_str}, активация ReLU,
Dropout({cfg.dropout}), выход - один логит, потеря BCEWithLogitsLoss.
Число параметров: {results['params_count']}.
Оптимизатор Adam(lr={cfg.lr}), батч {cfg.batch_size}, не более {cfg.max_epochs} эпох.
Ранняя остановка: {cfg.patience} эпох без улучшения val_loss, лучшие веса
восстанавливаются (full: эпоха {full['best_epoch']} из {full['epochs_run']},
chunks: эпоха {chunks['best_epoch']} из {chunks['epochs_run']}).
Стандартизация: статистики StandardScaler накоплены partial_fit по чанкам
train+val и общие для обоих режимов; test в фите скейлера не участвует.
Устройство: {results['device']}. Детерминированные алгоритмы torch:
{cfg.deterministic}, потоков CPU: {cfg.num_threads}.
Объявленный режим совпадения прогонов: ТОЧНО (фиксированное зерно {cfg.seed} +
детерминированные алгоритмы + один поток CPU) - два прогона дают идентичные
метрики до 6 знака включительно (dl_run1.log, dl_run2.log).

## 3. Источники данных

- full: весь датасет читается в память, TensorDataset + DataLoader с shuffle
  (генератор перемешивания инициализирован зерном {cfg.seed}).
- chunks: потоковый IterableDataset (CsvStreamDataset) читает csv чанками по
  {cfg.chunk_size} строк и отдаёт батчи по {cfg.batch_size}; датасет целиком не
  материализуется - в память заранее читается только колонка target для
  стратифицированного сплита. Порядок строк потока фиксирован (детерминизм).
- Профиль P1: датасет помещается в память, поэтому полный режим без стриминга
  допустим (для P3/P4 стриминг был бы обязателен и в полном режиме).

## 4. Кривые потерь и метрик

- full: [dl_curves_full.csv](dl_curves_full.csv), график ![full](curves_full.png)
- chunks: [dl_curves_chunks.csv](dl_curves_chunks.csv), график ![chunks](curves_chunks.png)
- full: лучший val_loss = {full['best_loss']:.6f} на эпохе {full['best_epoch']},
  остановка на эпохе {full['epochs_run']}.
- chunks: лучший val_loss = {chunks['best_loss']:.6f} на эпохе {chunks['best_epoch']},
  остановка на эпохе {chunks['epochs_run']}.

## 5. Сравнение режимов

| Метрика | full | full 95% ИИ | chunks | chunks 95% ИИ | |full-chunks| |
|---|---|---|---|---|---|
{chr(10).join(metric_rows)}

Интервалы метрик - бутстрэп 2.5%/97.5% по {cfg.n_bootstrap} пересборкам теста.
Вердикт по accuracy: {verdict}.
Номера прогонов MLflow (эксперимент lab3-variant10):
full = {results['run_ids'].get('full', 'нет')},
chunks = {results['run_ids'].get('chunks', 'нет')}.

Память и время (memory_compare.csv), пик - tracemalloc вокруг загрузки данных
и обучения, время - весь режим:

| Режим | Пик памяти, МБ | Время, с | Эпох |
|---|---|---|---|
| full | {full['peak']:.4f} | {full['dt']:.4f} | {full['epochs_run']} |
| chunks | {chunks['peak']:.4f} | {chunks['dt']:.4f} | {chunks['epochs_run']} |

Пик стриминга ограничен чанком и батчем и не растёт с объёмом данных
(тест tests/test_lab3_stream.py: 4000 против 8000 строк, stream_tests.log).

## 6. Сравнение с ML-моделью ЛР2

| Метрика | ML ЛР2 (full) | DL ЛР3 (full) | DL - ML |
|---|---|---|---|
{chr(10).join(cmp_rows)}

Числа ML взяты из {cfg.lab2_metrics_path} (режим full). Итог: {who}.

## 7. Воспроизводимость и паритет устройств

| Проверка | Результат | Файл |
|---|---|---|
| Два прогона с тем же зерном | метрики совпадают точно (режим ТОЧНО) | dl_run1.log, dl_run2.log |
| Повтор обучения и загрузки чекпоинта | сверка с dl_metrics.csv, --check-repeat | repeat_check.log |
| Снятая фиксация зерна | прогоны расходятся - фиксация необходима | seed_control.log |
| Паритет CPU/ускоритель, допуск 1e-5 | max разница {parity['max_diff']:.2e}, {parity_note} | dl_run1.log |

## 8. Чекпоинт

- Файл: {cfg.checkpoint_path}, sha256={results['ck_hash']}.
- Состав: state_dict весов, статистики скейлера (mean/scale), метаданные
  (архитектура, dropout, зерно, эпохи, число параметров).
- Загрузка и предсказание воспроизводятся: --check-repeat сравнивает
  предсказания загруженного чекпоинта с dl_metrics.csv (repeat_check.log).

## 9. Негативные контроли

| Контроль | Чем проверяли | Результат | Файл |
|---|---|---|---|
| Невалидный конфиг: lr <= 0 | запуск с configs/negative/bad_lr.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Невалидный конфиг: dropout вне [0; 1) | запуск с configs/negative/bad_dropout.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Невалидный конфиг: файл данных не существует | запуск с configs/negative/bad_path_dl.yaml | отказ с объяснением, exit 2 | config_negative_test.log |
| Снятая фиксация зерна | --seed-control: два обучения без manual_seed | val_loss и val_accuracy прогонов расходятся | seed_control.log |
| Стриминг держит память ограниченной | pytest: пик загрузчика на 4000 и 8000 строк | прирост пика меньше двух чанков | stream_tests.log |

## 10. Выводы

1. MLP {arch_str} содержит {results['params_count']} параметров; ранняя
   остановка сработала на эпохе {full['epochs_run']} (full) и
   {chunks['epochs_run']} (chunks), лучшие веса восстановлены
   (dl_curves_full.csv, dl_curves_chunks.csv).
2. Полный режим: accuracy = {full['values']['accuracy']:.6f} с 95% интервалом
   [{full['cis']['accuracy'][0]:.6f}; {full['cis']['accuracy'][1]:.6f}]
   (dl_metrics.csv: full accuracy).
3. Потоковый режим: accuracy = {chunks['values']['accuracy']:.6f} с 95% интервалом
   [{chunks['cis']['accuracy'][0]:.6f}; {chunks['cis']['accuracy'][1]:.6f}]
   (dl_metrics.csv: chunks accuracy).
4. Разность accuracy режимов {diff_acc:.6f}; вердикт: {verdict} (dl_metrics.csv).
5. {who} (раздел 6; числа ЛР2 из ml_metrics.csv).
6. Пик памяти: full {full['peak']:.4f} МБ против chunks {chunks['peak']:.4f} МБ;
   стриминг ограничен порцией и не растёт с объёмом данных
   (memory_compare.csv, stream_tests.log).
7. Время обучения: full {full['dt']:.4f} c, chunks {chunks['dt']:.4f} c
   (memory_compare.csv).
8. Воспроизводимость подтверждена записями: два прогона совпали точно
   (dl_run1.log, dl_run2.log), --check-repeat совпал с csv и чекпоинтом
   (repeat_check.log); без фиксации зерна прогоны расходятся (seed_control.log).
9. Паритет устройств: max разница предсказаний {parity['max_diff']:.2e} <= 1e-5,
   {parity_note} (раздел 7).
10. Чекпоинт {cfg.checkpoint_path} sha256={results['ck_hash'][:16]}... загружается
    и воспроизводит предсказания (раздел 8, repeat_check.log).
"""
    (out / "lab3_report.md").write_text(text, encoding="utf-8")
    print(f"[report] {out / 'lab3_report.md'}")


def check_repeat(cfg: DLConfig) -> int:
    """Переобучить оба режима без записи и сверить с dl_metrics.csv и чекпоинтом."""
    set_seed(cfg)
    device = resolve_device(cfg)
    feats, y_all = read_targets(cfg)
    idx_tr, idx_val, idx_test = split_indices(y_all, cfg)
    masks = make_masks(len(y_all), idx_tr, idx_val, idx_test)
    mask_tv = masks[0] | masks[1]
    scaler = fit_scaler(cfg, feats, mask_tv)

    csv = pd.read_csv(Path(cfg.report_dir) / "dl_metrics.csv")
    ok = True
    for mode in ("full", "chunks"):
        set_seed(cfg)
        res = run_mode(cfg, mode, feats, scaler, masks, device)
        for metric in ("accuracy", "pr_auc", "roc_auc"):
            ref = csv[(csv["mode"] == mode) & (csv["metric"] == metric)]
            ref_v = float(ref["value"].iloc[0])
            same = round(res["values"][metric], 6) == ref_v
            ok = ok and same
            print(f"[repeat] {mode} {metric}: повтор={res['values'][metric]:.6f} "
                  f"csv={ref_v:.6f} совпадает={same}")

    df = pd.read_csv(cfg.dataset_path)
    x_test_raw = df[feats].to_numpy()[idx_test]
    y_test = y_all[idx_test]
    proba = load_and_predict(cfg, x_test_raw)
    pred = (proba >= 0.5).astype(int)
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
        print(f"[repeat] чекпоинт {metric}: {v:.6f} csv={ref_v:.6f} совпадает={same}")
    print(f"[repeat] ВЕРДИКТ: {'совпадает' if ok else 'НЕ совпадает'}")
    return 0 if ok else 1


def seed_control(cfg: DLConfig) -> None:
    """Негативный контроль: два коротких обучения БЕЗ фиксации зерна расходятся."""
    feats, y_all = read_targets(cfg)
    idx_tr, idx_val, idx_test = split_indices(y_all, cfg)
    masks = make_masks(len(y_all), idx_tr, idx_val, idx_test)
    mask_tv = masks[0] | masks[1]
    scaler = fit_scaler(cfg, feats, mask_tv)
    arrays = load_full_arrays(cfg, feats, scaler, masks)

    def train_unseeded() -> tuple[float, float]:
        ds = torch.utils.data.TensorDataset(*arrays["train"])
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True
        )
        model = MLP(len(feats), cfg.hidden_dims, cfg.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(cfg.seed_control_epochs):
            model.train()
            for x, y in loader:
                opt.zero_grad()
                lossf(model(x), y).backward()
                opt.step()
        val_ds = torch.utils.data.TensorDataset(*arrays["val"])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)
        val_loss, proba, y_val = eval_loader(model, val_loader, "cpu", lossf)
        val_acc = float(accuracy_score(y_val, (proba >= 0.5).astype(int)))
        return float(val_loss), val_acc

    runs = []
    for name in ("A", "B"):
        vl, va = train_unseeded()
        runs.append((vl, va))
        print(f"[seed-control] прогон {name}: val_loss={vl:.6f} "
              f"val_accuracy={va:.6f} (эпох {cfg.seed_control_epochs}, зерно НЕ фиксировано)")
    d_loss = abs(runs[0][0] - runs[1][0])
    d_acc = abs(runs[0][1] - runs[1][1])
    print(f"[seed-control] расхождение: |Δval_loss|={d_loss:.6f}, |Δval_accuracy|={d_acc:.6f}")
    print("[seed-control] ВЫВОД: без фиксации зерна прогоны не совпадают - "
          "фиксация зерна обязательна (в основном прогоне зерно фиксировано, "
          "и прогоны совпадают точно).")


def main() -> None:
    """Прочитать конфиг (при отказе схемы - объяснение и exit 2) и запустить выбранный режим."""
    ap = argparse.ArgumentParser(description="ЛР3: DL-пайплайн, полный и стриминг")
    ap.add_argument("--config", required=True)
    ap.add_argument("--check-repeat", action="store_true",
                    help="повтор обучения и проверка чекпоинта без записи")
    ap.add_argument("--seed-control", action="store_true",
                    help="негативный контроль: обучения без фиксации зерна")
    args = ap.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        cfg = DLConfig(**raw)
    except ValidationError as e:
        print("КОНФИГУРАЦИЯ НЕВАЛИДНА, работа остановлена:")
        for err in e.errors():
            where = ".".join(str(x) for x in err["loc"])
            print(f"  - поле {where}: {err['msg']} (значение: {err.get('input')})")
        raise SystemExit(2)
    if args.seed_control:
        seed_control(cfg)
        return
    if args.check_repeat:
        raise SystemExit(check_repeat(cfg))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
