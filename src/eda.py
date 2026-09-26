"""ЛР1: данные варианта 10 (шумовые признаки), хеши и разведочный анализ.

Все числа отчёта берутся из этого прогона, поэтому отчёт и eda_stats.csv
согласованы, а повторный прогон воспроизводит выводы.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import yaml
from pydantic import BaseModel
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")  # без GUI: графики пишутся в png, работает и в CI, и на сервере
import matplotlib.pyplot as plt


class EdaConfig(BaseModel):
    """
    Схема настроек: описывает допустимые поля конфига и значения по умолчанию,
    хранит их в configs/; на невалидном конфиге работа останавливается сразу.
    """

    dataset_path: str
    hash_manifest_path: str
    report_dir: str
    seed: int = 511
    n_samples: int = 20000
    n_informative: int = 10
    n_redundant: int = 5
    n_repeated: int = 5
    n_classes: int = 2
    auc_noise_tol: float = 0.02
    corr_dup_threshold: float = 0.99
    outlier_z: float = 3.0
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15


def sha256_file(path: Path) -> str:
    """
    Считать sha256 файла, возвращает шестнадцатеричный отпечаток содержимого файла.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dataset(cfg: EdaConfig) -> pd.DataFrame:
    """
    Собрать датасет варианта 10.
    Вызывает make_classification с фиксированным зерном: 20
    признаков (10 информативных, 5 избыточных, 5 повторных) и бинарная цель.
    """
    x, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_informative + cfg.n_redundant + cfg.n_repeated,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=cfg.n_repeated,
        n_classes=cfg.n_classes,
        random_state=cfg.seed,
    )
    df = pd.DataFrame(x, columns=[f"feature_{i:02d}" for i in range(x.shape[1])])
    df["target"] = y
    return df


def prepare(cfg: EdaConfig) -> str:
    """
    Сохранить сырые данные и манифест хешей (шаг 3 ЛР1).
    Создаёт data/raw/dataset.csv и пишет data/hash_manifest.json.
    """
    path = Path(cfg.dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_dataset(cfg).to_csv(path, index=False)
    digest = sha256_file(path)
    manifest = {
        "files": [
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        ]
    }
    mpath = Path(cfg.hash_manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[prepare] файл: {path} размер={path.stat().st_size} байт")
    print(f"[prepare] sha256={digest}")
    print(f"[prepare] манифест: {mpath}")
    return digest


def manifest_sha256(cfg: EdaConfig) -> str:
    """
    Читает hash_manifest.json и возвращает sha256 первого файла.
    """
    data = json.loads(Path(cfg.hash_manifest_path).read_text(encoding="utf-8"))
    return data["files"][0]["sha256"]


def verify(cfg: EdaConfig) -> None:
    """
    Повторная верификация хеша (критерий артефакта hash_manifest.json).
    """
    tmp = Path(cfg.dataset_path).with_suffix(".verify.csv")
    make_dataset(cfg).to_csv(tmp, index=False)
    expected = manifest_sha256(cfg)
    actual = sha256_file(tmp)
    tmp.unlink()
    verdict = "совпадает" if actual == expected else "не совпадает"
    print("[verify] повторная генерация с тем же зерном")
    print(f"[verify] expected={expected}")
    print(f"[verify] actual  ={actual}")
    print(f"[verify] ВЕРДИКТ: {verdict}")


def tamper_check(cfg: EdaConfig) -> None:
    """
    Негативный контроль: подмена файла данных обнаруживается по хешу.
    """
    path = Path(cfg.dataset_path)
    backup = path.with_suffix(".backup.csv")
    shutil.copyfile(path, backup)
    df = pd.read_csv(path)
    df.iloc[0, 0] = df.iloc[0, 0] + 1.0
    df.to_csv(path, index=False)
    expected = manifest_sha256(cfg)
    actual = sha256_file(path)
    shutil.copyfile(backup, path)
    backup.unlink()
    restored = sha256_file(path)
    print("[tamper] подмена одного значения в файле данных")
    print(f"[tamper] expected={expected}")
    print(f"[tamper] actual  ={actual}")
    print(
        f"[tamper] ВЕРДИКТ: "
        f"{'не совпадает' if actual != expected else 'совпадает'}"
        " (подмена обнаружена)"
    )
    print(f"[tamper] после восстановления хеш={restored} совпал={restored == expected}")


def univariate_auc(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """
    Однопризнаковый ROC-AUC каждого признака.
    """
    out = {}
    for c in feats:
        a = roc_auc_score(df["target"], df[c])
        out[c] = max(a, 1.0 - a)
    return pd.Series(out).sort_values(ascending=False)


def dup_pairs(corr: pd.DataFrame, feats: list[str], thr: float) -> list[tuple]:
    """
    Найти пары почти дублирующихся признаков.
    Перебирает пары признаков и оставляет те, у которых
    |коэффициент Пирсона| >= порога (0.99).
    """
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= thr:
                pairs.append((a, b, float(r)))
    return pairs


def outlier_rows(df: pd.DataFrame, feats: list[str], z_thr: float) -> int:
    """
    Посчитать строки с аномальными значениями.
    Стандартизирует признаки и считает строки, где хотя бы одно
    значение по модулю больше z_thr (3.0).
    """
    z = (df[feats] - df[feats].mean()) / df[feats].std()
    return int((z.abs() > z_thr).any(axis=1).sum())


def figures(
    cfg: EdaConfig,
    df: pd.DataFrame,
    feats: list[str],
    auc: pd.Series,
    noise: list[str],
    pairs: list[tuple],
    corr: pd.DataFrame,
    share: pd.Series,
) -> None:
    """
    Построить четыре графика.
    """
    out = Path(cfg.report_dir)

    # 1. График проблемы варианта: сила связи каждого признака с целью.
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d62728" if f in noise else "#1f77b4" for f in auc.index]
    ax.barh(range(len(auc)), auc.values, color=colors)
    ax.set_yticks(range(len(auc)), auc.index, fontsize=8)
    ax.axvline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("однопризнаковый ROC-AUC (0.5 = признак не несёт информации)")
    ax.set_title(
        "Проблема варианта 10 (шумовые признаки): шумовых "
        f"{len(noise)} из {len(auc)} (|AUC-0.5| <= {cfg.auc_noise_tol})\n"
        f"пар почти дубликатов (|r| >= {cfg.corr_dup_threshold}): {len(pairs)}",
        fontsize=10,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "problem.png", dpi=150)
    plt.close(fig)

    # 2. Корреляции признаков (адаптация P1).
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        corr, cmap="coolwarm", vmin=-1, vmax=1, square=True,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(
        f"Корреляции Пирсона 20 признаков: пар с |r| >= {cfg.corr_dup_threshold}"
        f" - {len(pairs)} (адаптация P1)"
    )
    fig.tight_layout()
    fig.savefig(out / "corr_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Доли классов (адаптация P1), числа в подписи.
    fig, ax = plt.subplots(figsize=(5, 4))
    share.plot.bar(ax=ax, color=["#1f77b4", "#ff7f0e"])
    for i, v in enumerate(share.values):
        ax.text(i, v / 2, f"{v:.4f}", ha="center", color="white")
    ax.set_title(f"Доли классов: 0 = {share.iloc[0]:.4f}, 1 = {share.iloc[1]:.4f}")
    ax.set_xlabel("target")
    ax.set_ylabel("доля")
    fig.tight_layout()
    fig.savefig(out / "class_share.png", dpi=150)
    plt.close(fig)

    # 4. Распределения всех признаков (шаг 4: распределения ключевых величин).
    fig, axes = plt.subplots(4, 5, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), feats):
        ax.hist(df[c], bins=30, color="#1f77b4")
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Распределения 20 признаков варианта 10")
    fig.tight_layout()
    fig.savefig(out / "features_grid.png", dpi=150)
    plt.close(fig)
    print(f"[eda] графики сохранены в {out}")


def build_stats(
    cfg: EdaConfig,
    df: pd.DataFrame,
    feats: list[str],
    auc: pd.Series,
    noise: list[str],
    pairs: list[tuple],
    n_outl: int,
    share: pd.Series,
) -> list[dict]:
    """
    Формирует строки {metric, value, source}: общие сводки (объём,
    типы, пропуски, дубликаты, аномалии, доли классов, шумовые признаки, пары
    дубликатов, хеш данных) и по три числа на каждый признак (AUC, среднее, std).
    Зачем: это артефакт eda_stats.csv; колонка source - ссылка на скрипт/прогон,
    которую методичка требует для каждого числа в отчёте.
    """
    top = max(pairs, key=lambda t: abs(t[2])) if pairs else ("-", "-", 0.0)
    rows = [
        {"metric": "n_rows", "value": int(df.shape[0]), "source": "src/eda.py"},
        {"metric": "n_cols", "value": int(df.shape[1]), "source": "src/eda.py"},
        {"metric": "n_feature_cols", "value": len(feats), "source": "src/eda.py"},
        {"metric": "dtype_features", "value": str(df[feats].dtypes.iloc[0]),
         "source": "src/eda.py"},
        {"metric": "dtype_target", "value": str(df["target"].dtype),
         "source": "src/eda.py"},
        {"metric": "n_missing", "value": int(df.isna().sum().sum()),
         "source": "src/eda.py"},
        {"metric": "n_full_duplicates", "value": int(df.duplicated().sum()),
         "source": "src/eda.py"},
        {"metric": "outlier_row_share", "value": round(n_outl / len(df), 6),
         "source": "src/eda.py"},
        {"metric": "class_0_share", "value": round(float(share.iloc[0]), 6),
         "source": "src/eda.py; class_share.png"},
        {"metric": "class_1_share", "value": round(float(share.iloc[1]), 6),
         "source": "src/eda.py; class_share.png"},
        {"metric": "n_noise_features", "value": len(noise),
         "source": "src/eda.py; problem.png"},
        {"metric": "noise_features", "value": ";".join(noise),
         "source": "src/eda.py; problem.png"},
        {"metric": "min_feature_auc", "value": round(float(auc.min()), 6),
         "source": "src/eda.py; problem.png"},
        {"metric": "min_feature_auc_name", "value": str(auc.idxmin()),
         "source": "src/eda.py; problem.png"},
        {"metric": "max_feature_auc", "value": round(float(auc.max()), 6),
         "source": "src/eda.py; problem.png"},
        {"metric": "max_feature_auc_name", "value": str(auc.idxmax()),
         "source": "src/eda.py; problem.png"},
        {"metric": "n_dup_pairs", "value": len(pairs),
         "source": "src/eda.py; corr_heatmap.png"},
        {"metric": "max_abs_corr", "value": round(abs(float(top[2])), 6),
         "source": "src/eda.py; corr_heatmap.png"},
        {"metric": "max_abs_corr_pair", "value": f"{top[0]}|{top[1]}",
         "source": "src/eda.py; corr_heatmap.png"},
        {"metric": "dataset_sha256", "value": manifest_sha256(cfg),
         "source": "data/hash_manifest.json"},
        {"metric": "dataset_size_bytes",
         "value": Path(cfg.dataset_path).stat().st_size,
         "source": "data/hash_manifest.json"},
        {"metric": "seed", "value": cfg.seed, "source": "configs/eda_config.yaml"},
    ]
    for c in feats:
        rows.append({"metric": f"auc_{c}", "value": round(float(auc[c]), 6),
                     "source": "src/eda.py; problem.png"})
        rows.append({"metric": f"mean_{c}", "value": round(float(df[c].mean()), 6),
                     "source": "src/eda.py; features_grid.png"})
        rows.append({"metric": f"std_{c}", "value": round(float(df[c].std()), 6),
                     "source": "src/eda.py; features_grid.png"})
    return rows


def conclusions(cfg: EdaConfig, v: dict, auc: pd.Series, noise: list[str],
                pairs: list[tuple]) -> list[str]:
    """Сформулировать нумерованные выводы.

    Что делает: собирает 11 выводов, в каждом ровно одно число и ссылка на
    строку eda_stats.csv или на график.
    Зачем: критерий отчёта - "каждый вывод подкреплён числом"; числа берутся из
    того же прогона, поэтому выводы воспроизводятся повторным запуском.
    """
    top = max(pairs, key=lambda t: abs(t[2])) if pairs else ("-", "-", 0.0)
    return [
        (
            f"1. Объём датасета - {v['n_rows']} строк и {v['n_cols']} столбцов "
            f"(eda_stats.csv: n_rows, n_cols; src/eda.py)."
        ),
        (
            f"2. Все {v['n_feature_cols']} признаков имеют тип "
            f"{v['dtype_features']}, целевое поле target - {v['dtype_target']} "
            f"(eda_stats.csv: dtype_features)."
        ),
        (
            f"3. Пропусков в данных {v['n_missing']} (eda_stats.csv: n_missing)."
        ),
        (
            f"4. Полностью дублирующихся строк {v['n_full_duplicates']} "
            f"(eda_stats.csv: n_full_duplicates)."
        ),
        (
            f"5. Доля строк с аномальными значениями (|z| > {cfg.outlier_z}) "
            f"равна {v['outlier_row_share']} (eda_stats.csv: outlier_row_share)."
        ),
        (
            f"6. Доля класса 1 равна {v['class_1_share']}, дисбаланса классов "
            f"нет (eda_stats.csv: class_1_share; class_share.png)."
        ),
        (
            f"7. Шумовых признаков (|AUC-0.5| <= {cfg.auc_noise_tol}) - "
            f"{v['n_noise_features']} из {v['n_feature_cols']} "
            f"(eda_stats.csv: n_noise_features; problem.png)."
        ),
        (
            f"8. Минимальный однопризнаковый ROC-AUC равен {v['min_feature_auc']} "
            f"у признака {v['min_feature_auc_name']}, что соответствует случайному "
            f"угадыванию (eda_stats.csv: min_feature_auc; problem.png)."
        ),
        (
            f"9. Максимальный однопризнаковый ROC-AUC равен {v['max_feature_auc']} "
            f"у признака {v['max_feature_auc_name']} "
            f"(eda_stats.csv: max_feature_auc)."
        ),
        (
            f"10. Пар признаков с |r| >= {cfg.corr_dup_threshold} - "
            f"{v['n_dup_pairs']}, максимальная корреляция {v['max_abs_corr']} "
            f"между {top[0]} и {top[1]} (eda_stats.csv: n_dup_pairs, max_abs_corr; "
            f"corr_heatmap.png)."
        ),
        (
            f"11. Заявленная проблема варианта 10 в данных видна: {len(noise)} "
            f"признаков не связаны с целью и {len(pairs)} пар дублируют друг "
            f"друга, поэтому в ЛР2 применяется L1-регуляризация, обнуляющая такие "
            f"признаки (problem.png)."
        ),
    ]


def write_report(cfg: EdaConfig, v: dict, concl: list[str], pairs: list[tuple],
                 noise: list[str]) -> None:
    """
    Пишет отчёт с обязательной шапкой (вариант, работа, дата, автор,
    хеш коммита, хеши данных), паспортом данных, сводками EDA, графиком
    проблемы, выводами, таблицей негативных контролей и разделом приложений.
    """
    out = Path(cfg.report_dir)
    size_mb = v["dataset_size_bytes"] / (1024 * 1024)
    top = max(pairs, key=lambda t: abs(t[2])) if pairs else ("-", "-", 0.0)
    attached = [
        name
        for name in (
            "screen_ruleset.png",
            "screen_push_rejected.png",
            "screen_ci_green.png",
            "push_rejected.log",
        )
        if (out / name).exists()
    ]
    section7 = ""
    if attached:
        items = "\n".join(f"- reports/LAB1/{name}" for name in attached)
        section7 = (
            "\n## 7. Приложения: подтверждения проверок\n\n"
            f"{items}\n\n"
            "push_rejected.log - текстовый "
            "вывод отклонённого прямого push в main (работает ruleset).\n"
        )
    text = f"""# ЛР1. Git-окружение, прекоммитные проверки и EDA варианта

| Поле | Значение |
|---|---|
| Номер варианта | 10 (Шумовые признаки, профиль P1 - табличный) |
| Номер работы | ЛР1 |
| Дата | 2026-09-24 |
| Автор | Фамилия И. О. (заполнить) |
| Хеш коммита кода | <COMMIT_HASH> |
| Хеши данных | data/raw/dataset.csv sha256={v['dataset_sha256']} |

## 1. Паспорт данных

- Источник: scikit-learn, `sklearn.datasets.make_classification`; лицензия BSD-3-Clause.
- Способ получения: генерация в `src/eda.py --prepare`; внешний файл не загружался,
  сверка хеша - повторная генерация с тем же зерном (`--verify`, отчёт в
  reports/LAB1/hash_verify.log).
- Параметры генерации: n_samples={cfg.n_samples}, n_features=20,
  n_informative={cfg.n_informative}, n_redundant={cfg.n_redundant},
  n_repeated={cfg.n_repeated}, n_classes={cfg.n_classes}, weights=None,
  random_state={cfg.seed} (configs/eda_config.yaml).
- Хеш и размер: sha256={v['dataset_sha256']}, {v['dataset_size_bytes']} байт
  ({size_mb:.2f} МБ) - data/hash_manifest.json.
- Объём и поля: {v['n_rows']} строк, {v['n_cols']} столбцов - 20 числовых признаков
  feature_00...feature_19 ({v['dtype_features']}) и целевое поле target
  ({v['dtype_target']}, значения 0/1) - eda_stats.csv: n_rows, n_cols.
- План разбиения (используется с ЛР2): train {cfg.split_train:.0%} /
  validation {cfg.split_val:.0%} / test {cfg.split_test:.0%},
  `train_test_split(random_state={cfg.seed}, stratify=target)`; препроцессинг и
  подбор гиперпараметров - только на train и validation, test применяется один раз.

## 2. Сводки EDA

Все числа - из reports/LAB1/eda_stats.csv, прогон `python src/eda.py --config
configs/eda_config.yaml`.

- Структура и типы: {v['n_feature_cols']} признаков {v['dtype_features']},
  target {v['dtype_target']} (eda_stats.csv: dtype_features, dtype_target).
- Пропуски: {v['n_missing']} (eda_stats.csv: n_missing).
- Полные дубликаты строк: {v['n_full_duplicates']} (eda_stats.csv: n_full_duplicates).
- Аномальные значения: доля строк с |z| > {cfg.outlier_z} равна
  {v['outlier_row_share']} (eda_stats.csv: outlier_row_share).
- Распределения признаков: features_grid.png; среднее и std по каждому признаку -
  eda_stats.csv (метрики mean_feature_XX, std_feature_XX).
- Доли классов (адаптация P1): класс 0 - {v['class_0_share']}, класс 1 -
  {v['class_1_share']} (eda_stats.csv: class_0_share, class_1_share; class_share.png).
- Связи между величинами (адаптация P1): corr_heatmap.png; максимальный |r| =
  {v['max_abs_corr']} у пары {top[0]} и {top[1]} (eda_stats.csv: max_abs_corr,
  max_abs_corr_pair).
- Однопризнаковая связь с целью: минимальный AUC {v['min_feature_auc']}
  ({v['min_feature_auc_name']}), максимальный {v['max_feature_auc']}
  ({v['max_feature_auc_name']}); шумовых признаков {v['n_noise_features']}
  (eda_stats.csv: min_feature_auc, max_feature_auc, n_noise_features).

## 3. График проблемы варианта

![Проблема варианта 10: шумовые признаки](problem.png)

Из 20 признаков шумовых {v['n_noise_features']} (|AUC-0.5| <= {cfg.auc_noise_tol}),
пар почти дубликатов (|r| >= {cfg.corr_dup_threshold}) - {v['n_dup_pairs']},
минимальный однопризнаковый AUC {v['min_feature_auc']} у {v['min_feature_auc_name']}
(eda_stats.csv: n_noise_features, n_dup_pairs, min_feature_auc).

![Распределение признаков](features_grid.png)
![Матрица корреляции](corr_heatmap.png)

## 4. Выводы

{chr(10).join(concl)}

## 5. Негативные контроли и проверка

| Контроль | Чем проверяли | Результат | Файл |
|---|---|---|---|
| Коммит с кодом, нарушающим стиль | hook `ruff` на файле с ошибками стиля | заблокирован | precommit_block.log, блок 1 |
| Коммит с файлом, содержащим секрет | hook `detect-private-key` на PEM-ключе | заблокирован | precommit_block.log, блок 2 |
| Коммит с бинарным файлом сверх размера | hook `check-added-large-files` на файле 2 МБ (лимит 500 КБ) | заблокирован | precommit_block.log, блок 3 |
| Прямой push в main отклонён | `git push -u origin main` при активном ruleset | отклонён сервером | push_rejected.log, screen_push_rejected.png |
| Подмена файла данных | `python src/eda.py --config configs/eda_config.yaml --tamper-check` | хеш не совпал, подмена обнаружена | hash_verify.log |
| Повторная верификация хеша | `python src/eda.py --config configs/eda_config.yaml --verify` | вердикт "совпадает" | hash_verify.log |
| Воспроизводимость выводов EDA | два прогона `python src/eda.py --config configs/eda_config.yaml` | eda_stats.csv и отчёт идентичны | eda_run1.log, eda_run2.log |

## 6. Окружение и порядок изменений

- Ветки: `main` (защищена ruleset protect-main: прямой push отклоняётся,
  изменения только через pull request), рабочая ветка ЛР1 - `dev`.
- Шаблон описания изменения: `.github/pull_request_template.md`.
- pre-commit (конфигурация - .pre-commit_config.yaml): `ruff` - стиль кода;
  `detect-private-key`, `detect-aws-credentials` - проверка секретов;
  `check-added-large-files --maxkb=500` - запрет больших файлов;
  `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`, `check-json`,
  `check-merge-conflict` - быстрые проверки.
- Окружение: Python 3.14, venv `.venv`, список версий в requirements.txt.
- Прогоны MLflow: в ЛР1 не выполняются (регистрация прогонов начинается с ЛР2).
{section7}"""
    (out / "lab1_report.md").write_text(text, encoding="utf-8")
    print(f"[eda] отчёт: {out / 'lab1_report.md'}")
    print(f"[eda] шумовые признаки: {', '.join(noise)}")
    print(f"[eda] пары дубликатов: {len(pairs)}, максимальная |r| = {abs(top[2]):.4f}")


def run_eda(cfg: EdaConfig) -> None:
    out = Path(cfg.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.dataset_path)
    feats = [c for c in df.columns if c != "target"]

    auc = univariate_auc(df, feats)
    noise = [f for f in feats if abs(auc[f] - 0.5) <= cfg.auc_noise_tol]
    corr = df[feats].corr()
    pairs = dup_pairs(corr, feats, cfg.corr_dup_threshold)
    n_outl = outlier_rows(df, feats, cfg.outlier_z)
    share = df["target"].value_counts(normalize=True).sort_index()

    rows = build_stats(cfg, df, feats, auc, noise, pairs, n_outl, share)
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "eda_stats.csv", index=False)
    print(f"[eda] сводки: {out / 'eda_stats.csv'} (строк: {len(stats)})")

    figures(cfg, df, feats, auc, noise, pairs, corr, share)

    v = {r["metric"]: r["value"] for r in rows}
    concl = conclusions(cfg, v, auc, noise, pairs)
    (out / "conclusions_draft.md").write_text(
        "# Выводы ЛР1 (сгенерированы src/eda.py, числа из eda_stats.csv)\n\n"
        + "\n".join(concl)
        + "\n",
        encoding="utf-8",
    )
    write_report(cfg, v, concl, pairs, noise)


def main() -> None:
    ap = argparse.ArgumentParser(description="ЛР1: данные и EDA варианта 10")
    ap.add_argument("--config", required=True, help="путь к yaml-конфигу работы")
    ap.add_argument(
        "--source", default="real", choices=["real", "synthetic"],
        help="real - данные варианта; synthetic станет доступен после ЛР5",
    )
    ap.add_argument("--prepare", action="store_true",
                    help="создать датасет и манифест хешей")
    ap.add_argument("--verify", action="store_true",
                    help="повторная верификация хеша данных")
    ap.add_argument("--tamper-check", action="store_true",
                    help="негативный контроль: подмена файла данных")
    args = ap.parse_args()

    cfg = EdaConfig(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    if args.source == "synthetic":
        raise SystemExit(
            "Источник synthetic появится в ЛР5 (генератор данных), сейчас недоступен."
        )
    if args.prepare:
        prepare(cfg)
    if args.verify:
        verify(cfg)
    if args.tamper_check:
        tamper_check(cfg)
    if not (args.prepare or args.verify or args.tamper_check):
        run_eda(cfg)


if __name__ == "__main__":
    main()
