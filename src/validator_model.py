"""ЛР4: класс модели-валидатора.
1. Похожи ли новые данные на те, на которых модель училась (расстояние Махаланобиса).
2. Уверена ли сама модель в своем предсказании (порог достоверности).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dl_pipeline import MLP


def sha256(path):
    """Вспомогательная функция: считает хеш файла частями, чтобы не грузить его в память целиком."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ШАГ 1: Схема настроек валидатора (Pydantic)
class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    hash_manifest_path: str
    working_model_path: str  # путь к обученной модели из ЛР3 (.pt)
    report_dir: str

    features: list[str] = Field(min_length=1)

    # Способ оценки близости к обучающему распределению (по методичке для P1)
    proximity_method: str = "mahalanobis_per_class"

    seed: int = 511
    val_size: float = Field(default=0.15, gt=0, lt=1)
    test_size: float = Field(default=0.15, gt=0, lt=1)

    chunk_size: int = Field(default=2000, gt=0)  # Размер чанка для потокового чтения

    # Пороги отказа
    dist_quantile: float = Field(default=0.995, gt=0, lt=1)  # Отсекаем 0.5% самых странных "своих" данных
    conf_threshold: float = Field(default=0.65, gt=0.5, le=1)  # Если MLP уверена меньше чем на 65% — отказ

    # Целевые метрики качества валидатора (для проверки)
    recall_min: float = Field(default=0.95, ge=0, le=1)  # Полнота отказа чужеродных (должна быть >= 95%)
    false_reject_max: float = Field(default=0.05, ge=0, le=1)  # Доля ложных отказов своих (должна быть <= 5%)
    stat_tol: float = Field(default=1e-6, gt=0)  # Допуск совпадения статистик между full и chunks

    # Параметры генерации чужеродных примеров (для Шага 4)
    alien_size: int = Field(default=500, gt=0)
    shift_sigma: float = Field(default=3, gt=0)
    outlier_mult: float = Field(default=8, gt=0)
    foreign_scale: float = Field(default=40, gt=0)

    author: str = "Нефедова А.А."
    report_date: str

    @model_validator(mode="after")
    def check_settings(self):
        """Защита от невалидной конфигурации (негативный контроль)."""
        if self.val_size + self.test_size >= 1:
            raise ValueError("Сумма val_size и test_size должна быть меньше 1")

        if self.proximity_method != "mahalanobis_per_class":
            raise ValueError("Для P1 используйте mahalanobis_per_class")

        if len(set(self.features)) != len(self.features):
            raise ValueError("Имена признаков должны быть уникальны")

        for path in (self.dataset_path, self.hash_manifest_path, self.working_model_path):
            if not Path(path).is_file():
                raise ValueError(f"Файл не найден: {path}")

        return self


class DataValidator:
    """Валидатор на расстоянии Махаланобиса и уверенности MLP из ЛР3."""

    def __init__(self, cfg):
        self.cfg = cfg

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загружаем чекпоинт
        checkpoint = torch.load(cfg.working_model_path, map_location=device, weights_only=True)

        # Достаем метаданные (размерность входа, слои, dropout)
        meta = checkpoint["meta"]

        # Создаем экземпляр класса MLP, зная его параметры
        self.model = MLP(
            n_in=meta["n_feats"],
            hidden_dims=meta["hidden_dims"],
            dropout=meta["dropout"]
        )
        # Загружаем веса
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # Достаем параметры скейлера из чекпоинта
        self.mean = checkpoint["scaler_mean"].numpy()
        self.scale = checkpoint["scaler_scale"].numpy()

        if len(self.mean) != len(cfg.features) or np.any(self.scale <= 0):
            raise ValueError("Чекпоинт несовместим с признаками в конфиге")

        self.stats = None
        self.threshold = None
        self.conf = cfg.conf_threshold

    # ШАГ 3: Оценка статистик обучающего распределения (full и chunks)
    def fit(self, train_mask, mode):
        """Считаем средние, ковариации и диапазоны ТОЛЬКО на train (без утечек)."""
        if mode not in ("full", "chunks"):
            raise ValueError("Режим должен быть full или chunks")

        k = len(self.cfg.features)

        # Аккумуляторы для потокового подсчета статистик (формулы Уэлфорда)
        count = np.zeros(2, dtype=np.int64)  # Количество примеров каждого класса
        sums = np.zeros((2, k))  # Суммы значений по каждому классу
        cross = np.zeros((2, k, k))  # Матрицы перекрестных произведений для ковариации

        lo = np.full(k, np.inf)  # Минимумы по признакам (для диапазонов)
        hi = np.full(k, -np.inf)  # Максимумы по признакам

        # Читаем датасет либо целиком (full), либо кусками (chunks)
        reader = pd.read_csv(
            self.cfg.dataset_path,
            usecols=self.cfg.features + ["target"],
            chunksize=(self.cfg.chunk_size if mode == "chunks" else None),
        )

        blocks = reader if mode == "chunks" else [reader]

        for block in blocks:
            # Применяем маску train, чтобы случайно не заглянуть в val или test
            part = block.loc[train_mask[block.index.to_numpy()]]
            if part.empty:
                continue

            # Стандартизируем данные
            x = (part[self.cfg.features].to_numpy(dtype=np.float64) - self.mean) / self.scale
            y = part["target"].to_numpy()

            if not np.isfinite(x).all():
                raise ValueError("В обучающих данных есть неконечные значения")

            # Обновляем минимумы и максимумы (диапазоны)
            lo = np.minimum(lo, x.min(axis=0))
            hi = np.maximum(hi, x.max(axis=0))

            # Накапливаем статистики для каждого класса (0 и 1)
            for class_id in (0, 1):
                class_x = x[y == class_id]
                count[class_id] += len(class_x)
                sums[class_id] += class_x.sum(axis=0)
                cross[class_id] += class_x.T @ class_x

        if min(count) < 2:
            raise ValueError("В train должно быть не меньше двух строк каждого класса")

        # Считаем средние по классам
        mu = sums / count[:, None]

        # Считаем общую ковариационную матрицу (учитывает корреляции между признаками)
        cov = sum(
            cross[class_id] - count[class_id] * np.outer(mu[class_id], mu[class_id])
            for class_id in (0, 1)
        )
        cov /= count.sum() - 2

        # Регуляризация: добавляем "пылинку" к диагонали, чтобы матрицу можно было обратить.
        # Это спасает от ошибки, если в данных есть дублирующиеся признаки (как в твоем варианте 10).
        reg = cov + (1e-6 * max(np.trace(cov) / k, 1e-12) * np.eye(k))

        self.stats = {
            "mu": mu,  # Средние классов
            "cov": cov,  # Ковариация
            "inv": np.linalg.inv(reg),  # Обратная ковариация (для Махаланобиса)
            "lo": lo,  # Диапазоны (минимумы)
            "hi": hi,  # Диапазоны (максимумы)
            "count": count,  # Количество примеров
        }
        return self

    # ШАГ 2: Расчет расстояния Махаланобиса
    def distance(self, x):
        """Считаем расстояние от точки до ближайшего центра класса."""
        if self.stats is None:
            raise ValueError("Сначала оцените или загрузите статистики")

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Ожидаются конечные числовые признаки в правильном порядке")

        # Стандартизация
        z = (x - self.mean) / self.scale
        distances = []

        # Считаем расстояние до центров класса 0 и класса 1
        for center in self.stats["mu"]:
            diff = z - center

            # Формула Махаланобиса: (x - mu)^T * inv_cov * (x - mu)
            # einsum делает это быстро для всей пачки строк.
            square = np.einsum("ij,jk,ik->i", diff, self.stats["inv"], diff)

            distances.append(np.sqrt(np.maximum(square, 0)))

        # Возвращаем минимальное расстояние (если точка похожа хотя бы на один класс — она "своя")
        return np.min(distances, axis=0)

    # ШАГ 3: Калибровка порога на validation
    def calibrate(self, x_val):
        """Устанавливаем порог по квантилю (99.5%) расстояний на validation."""
        self.threshold = float(np.quantile(self.distance(x_val), self.cfg.dist_quantile))

        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Порог расстояния должен быть конечным и положительным")
        return self

    # ШАГ 2: Получение вероятности от MLP модели (из ЛР3)
    def probability(self, x):
        """Прогоняем данные через нейросеть, чтобы получить вероятность класса 1."""
        # Стандартизируем вход с использованием средних и масштабов
        h = torch.as_tensor((np.asarray(x, dtype=np.float64) - self.mean) / self.scale, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(h)
            # Применяем сигмоиду для получения вероятности
            return torch.sigmoid(logits).numpy().astype(np.float64)

    # ШАГ 2, 4: Правило отказа
    def decide(self, x, declared=None):
        """Решаем: пропустить или отклонить. Считаем непрерывный score для ROC-AUC."""
        if self.threshold is None:
            raise ValueError("Сначала установите или загрузите порог")

        distances = self.distance(x)
        probability = self.probability(x)

        # Считаем уверенность. Если метка передана, берем вероятность именно этого класса.
        # Если нет — берем максимальную вероятность (насколько модель уверена в своем выборе).
        if declared is None:
            confidence = np.maximum(probability, 1 - probability)
        else:
            declared = np.asarray(declared)
            if declared.shape != probability.shape or not np.isin(declared, [0, 1]).all():
                raise ValueError("Переданные метки должны быть 0/1 для каждой строки")
            confidence = np.where(declared == 1, probability, 1 - probability)

        # Считаем score (для ROC-AUC): чем больше, тем более плохие данные
        score = np.maximum(
            distances / self.threshold,
            self.conf / np.maximum(confidence, 1e-12),
        )

        # ГЛАВНОЕ ПРАВИЛО: отклоняем, если расстояние больше порога ИЛИ уверенность ниже порога
        rejected = (distances > self.threshold) | (confidence < self.conf)

        return rejected, score

    # ШАГ 5: Сохранение с хешем
    def save(self, path):
        """Сохраняем все параметры валидатора в NPZ файл."""
        if self.stats is None or self.threshold is None:
            raise ValueError("Сначала обучите и откалибруйте валидатор")

        np.savez(
            path,
            **self.stats,
            threshold=self.threshold,
            conf=self.conf,
            features=np.asarray(self.cfg.features),
            mean=self.mean,
            scale=self.scale,
            model_sha256=sha256(self.cfg.working_model_path),  # Привязываем к конкретной модели ЛР3
        )
        return sha256(path)

    # ШАГ 5: Загрузка с проверкой
    def load(self, path):
        """Загружаем валидатор и проверяем, что он не испорчен и привязан к правильной модели."""
        with np.load(path, allow_pickle=False) as data:
            # Проверяем, что признаки идут в том же порядке
            if list(data["features"]) != self.cfg.features:
                raise ValueError("Порядок признаков в NPZ не совпадает с конфигом")

            # Проверяем хеш модели ЛР3. Если модель переобучили — валидатор откажется работать.
            if str(data["model_sha256"]) != sha256(self.cfg.working_model_path):
                raise ValueError("Хеш рабочей модели не совпадает с NPZ")

            self.stats = {key: data[key] for key in ("mu", "cov", "inv", "lo", "hi", "count")}
            self.threshold = float(data["threshold"])
            self.conf = float(data["conf"])
            self.mean = data["mean"]
            self.scale = data["scale"]

        return self
