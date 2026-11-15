"""ЛР5: управляемый генератор табличных данных профиля P1.

Суть: мы создаём искусственные данные, которые статистически
похожи на реальные (режим similar), либо полный шум (режим random).
А ещё мы умеем намеренно "портить" данные (дрейф).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Настройки "ручек" дрейфа.
class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Сдвиг средних (в стандартных отклонениях). 0 - не сдвигаем, 2 - сдвинули сильно вправо.
    mean_shift: float = Field(default=0.0, ge=-3.0, le=3.0)

    # Доля класса 1. Если None - берём из реальных данных (обычно 0.5).
    # Если 0.1 - сделаем класс 1 редким (имитация дисбаланса).
    class_one_share: float | None = Field(default=None, ge=0.0, le=1.0)

    # Переворот меток. 0.1 - значит 10% правильных ответов станут неправильными.
    label_flip: float = Field(default=0.0, ge=0.0, le=1.0)

    # Амплитуда сезонной волны (синусоиды) в стандартных отклонениях.
    season_amplitude: float = Field(default=0.0, ge=0.0, le=3.0)
    # Период волны (каждые N строк значения будут подниматься и опускаться).
    season_period: int = Field(default=20, ge=2)


# Общие настройки генератора.
class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str  # "similar" (похожие) или "random" (шум)
    n_rows: int = Field(gt=0)  # Сколько строк создать
    seed: int = Field(ge=0)  # Зерно для воспроизводимости
    features: list[str] = Field(min_length=1)  # Список колонок с признаками
    target: str = "target"  # Имя целевой колонки
    drift: DriftConfig = Field(default_factory=DriftConfig)  # Вложенные настройки дрейфа

    @model_validator(mode="after")
    def check_schema(self):
        """Отклоняем неизвестный режим, дубли и пересечение с целью."""
        # Режим должен быть строго "similar" или "random"
        if self.mode not in ("similar", "random"):
            raise ValueError("mode должен быть similar или random")

        # Нельзя, чтобы признаки дублировались или в них было имя target
        if (len(set(self.features)) != len(self.features) or self.target in self.features):
            raise ValueError("Признаки должны быть уникальны и не включать target")

        # Нельзя пустые имена колонок
        if any(not name.strip() for name in [*self.features, self.target]):
            raise ValueError("Имена столбцов не должны быть пустыми")

        return self


# Сам класс генератора.
class SyntheticGenerator:
    """Генерируем строки в числовых диапазонах обучающей выборки."""

    def __init__(self, cfg: GeneratorConfig, reference: pd.DataFrame):
        """При создании класса мы изучаем реальные данные (reference),
        чтобы знать их границы и пропорции."""
        self.cfg = cfg
        required = cfg.features + [cfg.target]

        # Проверяем, что в реальных данных есть все нужные колонки и они не пустые
        if (any(name not in reference.columns for name in required) or reference.empty):
            raise ValueError("Нет обязательных столбцов или обучающая таблица пуста")

        # Запоминаем эталонный датасет (только нужные колонки)
        self.reference = reference.loc[:, required].copy().reset_index(drop=True)

        # Достаём признаки (X) в виде чисел и метки (Y)
        self.values = self.reference[cfg.features].to_numpy(dtype=np.float64)
        self.labels = self.reference[cfg.target].to_numpy()

        # Данные должны быть нормальными числами, а метки - строго 0 и 1
        if not np.isfinite(self.values).all():
            raise ValueError("Признаки должны быть конечными числами")
        if set(np.unique(self.labels)) != {0, 1}:
            raise ValueError("В обучающей таблице должны быть классы 0 и 1")

        # Запоминаем границы реальных данных (минимум, максимум, разброс)
        self.minimum = self.values.min(axis=0)
        self.maximum = self.values.max(axis=0)
        self.std = self.values.std(axis=0)
        self.share = float(np.mean(self.labels == 1))  # Изначальная доля класса 1


    # Режим "похожие".
    def _similar(self, rng, labels):
        """Берём целые обучающие строки внутри каждого класса."""
        # Создаём пустую заготовку под результат
        result = np.empty((len(labels), len(self.cfg.features)), dtype=np.float64)

        # Для каждого класса (0 и 1) отдельно выдёргиваем строки из реальных данных
        for label in (0, 1):
            positions = np.flatnonzero(labels == label)
            pool = np.flatnonzero(self.labels == label)

            # выборка с возвращением
            result[positions] = self.values[rng.choice(pool, size=len(positions), replace=True)]

        return result


    # Режим "случайные" (Полный шум).
    def _random(self, rng, n_rows):
        """Генерируем признаки равномерно в допустимых диапазонах (от min до max)."""
        return rng.uniform(self.minimum, self.maximum, size=(n_rows, len(self.cfg.features)))

    # Применение дрейфа.
    def _apply_drift(self, values):
        """Сдвигаем средние и добавляем периодическую волну."""
        drift = self.cfg.drift

        # Считаем фазу синусоиды
        phase = (2 * np.pi * np.arange(len(values)) / drift.season_period)

        # Сама волна: амплитуда * sin(фаза) * разброс_признака
        wave = (drift.season_amplitude * np.sin(phase)[:, None] * self.std)

        # Прибавляем сдвиг средних и волну
        shifted = values + drift.mean_shift * self.std + wave

        # Обрезаем, чтобы не выйти за пределы реальных данных
        return np.clip(shifted, self.minimum, self.maximum)

    # Главный метод генерации.
    def generate(self):
        """Возвращаем DataFrame по схеме features + target."""
        cfg = self.cfg
        # Создаём генератор случайных чисел с фиксированным зерном
        rng = np.random.default_rng(cfg.seed)

        # Определяем долю единиц: либо из реальных данных, либо из ручки дрейфа
        share = self.share if cfg.drift.class_one_share is None else cfg.drift.class_one_share

        labels = (rng.random(cfg.n_rows) < share).astype(np.int64)

        # Генерируем сами признаки в зависимости от выбранного режима
        if cfg.mode == "similar":
            values = self._similar(rng, labels)
        else:
            values = self._random(rng, cfg.n_rows)

        # Портим данные с помощью дрейфа (сдвиг, сезонность)
        values = self._apply_drift(values)

        # Генерируем случайные числа и смотрим, какие из них меньше порога label_flip.
        flipped = rng.random(cfg.n_rows) < cfg.drift.label_flip

        # Там, где сработал переворот, меняем метку: 1 -> 0, 0 -> 1
        labels = np.where(flipped, 1 - labels, labels)

        # Собираем всё обратно в DataFrame
        result = pd.DataFrame(values, columns=cfg.features)
        result[cfg.target] = labels

        return result

    # Фиксация формата CSV для побайтовой проверки.
    def csv_bytes(self):
        """Возвращаем UTF-8 CSV с фиксированным форматом чисел."""
        stream = io.StringIO(newline="")

        # to_csv с float_format="%.10f" гарантирует, что число 0.1 всегда запишется
        # как "0.1000000000", а не "0.1" или "0.10000000000000001".
        # Это нужно, чтобы хеши файлов совпадали при повторных запусках.
        self.generate().to_csv(
            stream,
            index=False,
            float_format="%.10f",
            lineterminator="\n",
        )

        return stream.getvalue().encode("utf-8")

    def save_csv(self, path):
        """Сохраняем результат для использования в следующих работах."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Пишем байты напрямую, чтобы избежать проблем с кодировками Windows
        path.write_bytes(self.csv_bytes())

        return path
