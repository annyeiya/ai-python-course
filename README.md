# AI Python Course
# Лабораторные работы, вариант 10 («Шумовые признаки», профиль P1 — табличный)


## Где что лежит

```
configs/
  eda_config.yaml          схема настроек ЛР1 (генерация и пороги EDA)
  ml_config.yaml           схема настроек ЛР2 (pydantic MLConfig)
  dl_config.yaml           схема настроек ЛР3 (pydantic DLConfig)
  negative/                невалидные конфиги-контроли (ЛР2: 3 шт., ЛР3: 3 шт.)
src/
  eda.py                   ЛР1: генерация датасета, манифест хешей, verify,
                           tamper-check, EDA-сводки, графики, отчёт
  ml_pipeline.py           ЛР2: sklearn (StandardScaler + LogReg L1 saga /
                           SGD partial_fit), бутстрэп, MLflow, отчёты
  dl_pipeline.py           ЛР3: torch MLP (full / потоковый IterableDataset),
                           ранняя остановка, паритет устройств, MLflow, отчёты
tests/
  test_lab2_leak.py        ЛР2: ловушка утечки + память чанков
  test_lab3_stream.py      ЛР3: память стриминга + отказы схемы + фиксация зерна
data/
  raw/dataset.csv          данные варианта (в git НЕ входят, только хеш)
  hash_manifest.json       sha256 данных (в git)
reports/
  LAB1/  lab1_report.md, eda_stats.csv, corr_heatmap.png, features_grid.png, problem.png
  LAB2/  lab2_report.md, ml_metrics.csv, ml_model.joblib,
         логи: ml_run1/2, repeat_check, leak_memory_tests, config_negative_test
  LAB3/  lab3_report.md, dl_metrics.csv, memory_compare.csv, dl_model.pt,
         dl_curves_*.csv + curves_*.png,
         логи: dl_run1/2, repeat_check, seed_control, stream_tests, config_negative_test
```


```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Воспроизведение ЛР1 (данные и EDA)

```bash
python3 src/eda.py --config configs/eda_config.yaml --prepare
python3 src/eda.py --config configs/eda_config.yaml --verify
python3 src/eda.py --config configs/eda_config.yaml --tamper-check
python3 src/eda.py --config configs/eda_config.yaml
```

## Воспроизведение ЛР2 (ML-пайплайн, full против чанков)

```bash
python3 src/ml_pipeline.py --config configs/ml_config.yaml
python3 src/ml_pipeline.py --config configs/ml_config.yaml
python3 -m pytest tests -q
python3 src/ml_pipeline.py --config configs/negative/bad_chunk.yaml
python3 src/ml_pipeline.py --config configs/ml_config.yaml --check-repeat
```

## Воспроизведение ЛР3 (DL-пайплайн, full против стриминга)

Вариант A — локально (CPU):
```bash
python3 src/dl_pipeline.py --config configs/dl_config.yaml
python3 src/dl_pipeline.py --config configs/dl_config.yaml
python3 -m pytest tests -q
python3 src/dl_pipeline.py --config configs/negative/bad_chunk.yaml
python3 src/dl_pipeline.py --config configs/dl_config.yaml --check-repeat
python3 src/dl_pipeline.py --configs/dl_config.yaml --seed-control
```

Вариант B — Colab (GPU, настоящий паритет CPU/CUDA): на Google Drive
копируются только 6 файлов (`src/dl_pipeline.py`, `configs/dl_config.yaml`, `data/raw/dataset.csv`, `data/hash_manifest.json`,
`reports/LAB2/ml_metrics.csv`), в ноутбуке: runtime T4 GPU →
`pip install mlflow==3.17.2` → копирование папки в `/content` → прогоны 1–2,
`--seed-control`, `--check-repeat` → `zip` папки `reports/LAB3` и скачивание.
Тесты и негативные конфиги выполняются локально в WSL (от устройства не зависят).
После GPU-прогона пайплайн в WSL не перезапускать: артефакты должны происходить
из одной среды.
