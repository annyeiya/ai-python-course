"""ЛР6: FastAPI-сервис прогнозов и атомарного переключения моделей."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Архитектура берётся из ЛР3; заново MLP не описываем.
from dl_pipeline import MLP
from validator_model import DataValidator

# Одна строка из 20 конечных числовых признаков
# strict=True запрещает приводить строки к float, allow_inf_nan=False
# отсекает NaN/Inf ещё на уровне схемы.
Feature = Annotated[
    float,
    Field(strict=True, ge=-1000, le=1000, allow_inf_nan=False),
]


class PredictRequest(BaseModel):
    # extra="forbid" — лишние поля в JSON дают 422, а не молча игнорируются.
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=100)
    features: list[Feature] = Field(min_length=20, max_length=20)


class ValidationResult(BaseModel):
    """Предупреждение — это оценка, а не запрет на прогноз."""

    suspicious: bool
    score: float


class PredictResponse(BaseModel):
    request_id: str
    version: str
    prediction: int
    probability: float
    validation: ValidationResult


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)


def load_verified(entry):
    """Шаг 3: хешируем и загружаем одни и те же байты модели.

    Важно: хеш считается по тем байтам, которые потом и десериализуются
    (через io.BytesIO), иначе можно проверить один файл, а загрузить другой.
    """
    raw = Path(entry["path"]).read_bytes()

    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("SHA-256 модели не совпадает с реестром")

    if entry["kind"] == "sklearn":
        # Joblib загружаем только из доверенного источника.
        model = joblib.load(io.BytesIO(raw))
        return ("sklearn", model)

    if entry["kind"] == "torch":
        checkpoint = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )

        meta = checkpoint["meta"]

        # Создаём архитектуру через класс из dl_pipeline.
        model = MLP(
            meta["n_feats"],
            meta["hidden_dims"],
            meta["dropout"],
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return (
            "torch",
            model,
            checkpoint["scaler_mean"].numpy(),
            checkpoint["scaler_scale"].numpy(),
        )

    raise ValueError("Неизвестный тип модели")


def predict(model, features):
    """Получаем вероятность класса 1 от модели ЛР2 или ЛР3."""
    x = np.asarray([features], dtype=np.float64)

    if model[0] == "sklearn":
        # Масштабирование находится внутри pipeline ЛР2.
        return float(model[1].predict_proba(x)[0, 1])

    # Распаковка torch-варианта: сеть + параметры стандартизации.
    _, net, mean, scale = model

    tensor = torch.from_numpy(
        ((x - mean) / scale).astype(np.float32)
    )

    with torch.no_grad():  # не строим граф вычислений на инференсе
        return float(torch.sigmoid(net(tensor))[0].item())


class Registry:
    """Шаги 3–4: активная модель и история внутри одного процесса."""

    def __init__(self, path):
        self.path = Path(path)

        self.data = json.loads(
            self.path.read_text(encoding="utf-8")
        )

        # lock — быстрый доступ к версии/модели и данным реестра;
        # transition_lock — сериализует тяжёлые переключения, чтобы кандидаты не загружались параллельно.
        self.lock = threading.RLock()
        self.transition_lock = threading.Lock()

        self.version = self.data["active"]
        self.model = load_verified(
            self.data["versions"][self.version]
        )

        # Проверяем прогноз до приёма запросов (fail fast при старте).
        predict(self.model, [0.0] * 20)

    def snapshot(self):
        """Запрос получает согласованную пару версия/модель.

        Оба поля читаются под одним локом, поэтому нельзя получить
        версию от одной модели, а объект — от другой.
        """
        with self.lock:
            return self.version, self.model

    def describe(self):
        """Возвращаем копию реестра."""
        with self.lock:
            # deepcopy — чтобы вызывающий код не мог случайно изменить внутреннее состояние реестра.
            return copy.deepcopy(self.data)

    def _write(self, value):
        """Заменяем реестр через временный файл."""
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ) + "\n"

        # Атомарная замена: пишем во временный файл в той же директории
        # (важно для os.replace на одной ФС), fsync, затем os.replace.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
        ) as file:
            temp = file.name

            try:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())  # данные реально на диске
            except BaseException:
                os.unlink(temp)
                raise

        try:
            os.replace(temp, self.path)
        finally:
            # Если replace не удался — не оставляем мусорный temp-файл.
            if os.path.exists(temp):
                os.unlink(temp)

    def switch(self, target, reason, rollback=False):
        """Загружаем кандидата до замены активной модели."""
        with self.transition_lock:
            # Читаем запись кандидата под коротким локом и сразу отпускаем:
            # тяжёлую загрузку делаем без блокировки запросов.
            with self.lock:
                if target not in self.data["versions"]:
                    raise KeyError("Версия отсутствует в реестре")

                entry = copy.deepcopy(
                    self.data["versions"][target]
                )

            # Пока кандидат загружается, запросы идут на старой модели.
            candidate = load_verified(entry)
            predict(candidate, [0.0] * 20)  # smoke-тест кандидата

            with self.lock:
                old = self.version

                if old == target:
                    # Идемпотентность: повторное продвижение той же версии.
                    return {
                        "active": old,
                        "changed": False,
                    }

                updated = copy.deepcopy(self.data)
                now = datetime.now(UTC).isoformat()

                # Обновляем статусы и историю обеих версий.
                for version, status in (
                    (old, "снята"),
                    (target, "рабочая"),
                ):
                    item = updated["versions"][version]
                    item["status"] = status

                    item["history"].append(
                        {
                            "at": now,
                            "status": status,
                            "reason": reason,
                        }
                    )

                updated["active"] = target

                # При откате не перезаписываем previous, иначе
                # повторный rollback вернёт ту же версию (зацикливание).
                if not rollback:
                    updated["previous"] = old

                self._write(updated)     # сначала фиксируем на диске
                self.data = updated      # затем обновляем память
                self.version = target
                self.model = candidate   # и только потом активную модель

                return {
                    "active": target,
                    "changed": True,
                }

    def rollback(self):
        """Возвращаем версию до последнего продвижения."""
        with self.lock:
            target = self.data.get("previous")

        if target is None:
            raise ValueError("Нет предыдущей версии для отката")

        return self.switch(
            target,
            "откат на предыдущую версию",
            rollback=True,
        )


# Один поток на инференс: предсказуемая нагрузка и меньше конкуренции.
torch.set_num_threads(1)

# state_lock защищает только счётчики stats.
state_lock = threading.Lock()

stats = {
    "requests": 0,
    "http_5xx": 0,
    "predictions": 0,
    "warnings": 0,
}

registry = None
validator = None


@asynccontextmanager
async def lifespan(_app):
    """Загружаем рабочую модель и валидатор при старте."""
    global registry, validator

    registry_path = os.environ.get(
        "LAB6_REGISTRY",
        "reports/LAB6/registry.json",
    )

    registry = Registry(registry_path)

    # Валидатор ЛР4 привязан к DL.
    entry = registry.describe()["versions"]["dl-v1"]

    digest = hashlib.sha256(
        Path(entry["path"]).read_bytes()
    ).hexdigest()

    if digest != entry["sha256"]:
        raise ValueError(
            "SHA-256 DL-модели для валидатора не совпадает с реестром"
        )

    # Для загрузки готового валидатора нужны только эти поля.
    cfg = SimpleNamespace(
        working_model_path=entry["path"],
        features=[f"feature_{i:02d}" for i in range(20)],
        conf_threshold=0.65,
    )

    # Порог из NPZ заменит временное значение conf_threshold.
    validator = DataValidator(cfg).load(
        "reports/LAB4/validator_params.npz"
    )

    # Smoke-тест валидатора: оценка должна быть конечной.
    _, scores = validator.decide(
        np.zeros((1, 20))
    )

    if not np.isfinite(scores[0]):
        raise ValueError(
            "Валидатор вернул неконечную оценку при запуске"
        )

    yield  # здесь приложение работает; после yield — код остановки


app = FastAPI(
    title="ЛР6: сервис модели P1",
    lifespan=lifespan,
)


@app.middleware("http")
async def count_requests(request, call_next):
    """Считаем запросы и серверные ошибки."""
    # 500 по умолчанию: если call_next бросит исключение,
    # в stats попадёт именно серверная ошибка.
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # finally — чтобы счётчики обновились и при исключении.
        with state_lock:
            stats["requests"] += 1

            if status >= 500:
                stats["http_5xx"] += 1


# Адреса сервиса модели.

@app.get("/health")
def health():
    return {
        "status": "ok",
        "active": registry.snapshot()[0],
    }


@app.get("/models")
def models():
    # describe() отдаёт копию — наружу не утекает внутреннее состояние.
    return registry.describe()


@app.get("/metrics")
def metrics():
    with state_lock:
        result = dict(stats)  # снимаем согласованный снимок счётчиков

    result["active"] = registry.snapshot()[0]
    return result


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest):
    # Версия и объект модели фиксируются вместе.
    version, model = registry.snapshot()

    # Без метки: предупреждаем, но не отказываем в прогнозе.
    rejected, scores = validator.decide(
        np.asarray([body.features], dtype=np.float64)
    )

    suspicious = bool(rejected[0])
    score = float(scores[0])

    if not np.isfinite(score):
        raise ValueError("Валидатор вернул неконечную оценку")

    probability = predict(model, body.features)

    with state_lock:
        stats["predictions"] += 1
        stats["warnings"] += int(suspicious)

    return PredictResponse(
        request_id=body.request_id,
        version=version,
        prediction=int(probability >= 0.5),
        probability=probability,
        validation=ValidationResult(
            suspicious=suspicious,
            score=score,
        ),
    )


@app.post("/promote")
def promote(body: PromoteRequest):
    try:
        return registry.switch(
            body.version,
            "продвижение через API",
        )
    except (KeyError, ValueError, OSError) as exc:
        # Конфликт состояния/реестра — 409, а не 500.
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post("/rollback")
def rollback():
    try:
        return registry.rollback()
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
