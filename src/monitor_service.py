"""ЛР6: отдельный сервис мониторинга модели."""
import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import matplotlib

# Создаём графики без графического рабочего стола.
# Agg — неинтерактивный backend: обязателен на сервере без X11/дисплея.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# Мониторинг работает в отдельном процессе
# lock защищает словарь state от гонок между фоновым опросом (sample_loop) и обработчиками HTTP (/metrics).
lock = threading.Lock()

state = {
    "samples": [],   # последние наблюдения (для графика и /metrics)
    "polls": 0,      # сколько всего опросов модели сделано
    "failures": 0,   # сколько из них завершились ошибкой
}

# Каталог, куда пишем график и jsonl-журнал.
out = Path(
    os.environ.get(
        "LAB6_MONITOR_DIR",
        "reports/LAB6",
    )
)

# Адрес сервиса модели, который опрашиваем.
model_url = os.environ.get(
    "LAB6_MODEL_URL",
    "http://127.0.0.1:8001",
)


def render(samples):
    """Сохраняем график состояния, прогнозов и предупреждений."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 5),
        sharex=True,  # общая ось X — наблюдения синхронны на обоих графиках
    )

    x = list(range(len(samples)))

    # Верхний график: доступность модели (1/0).
    axes[0].step(
        x,
        [int(sample["ok"]) for sample in samples],
        where="post",
        label="модель",
    )

    # Линия «монитор» всегда 1 — визуальный ориентир «мы живы».
    axes[0].plot(
        x,
        [1] * len(samples),
        linestyle="--",
        label="монитор",
    )

    axes[0].legend()
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_ylabel("health: 0/1")

    # Нижний график: накопленные счётчики из /metrics модели.
    axes[1].plot(
        x,
        [sample.get("predictions", 0) for sample in samples],
        label="прогнозы",
    )

    axes[1].plot(
        x,
        [sample.get("warnings", 0) for sample in samples],
        label="предупреждения",
    )

    axes[1].legend()
    axes[1].set_ylabel("количество")
    axes[1].set_xlabel("номер наблюдения")

    fig.tight_layout()

    # Пишем во временный файл и атомарно подменяем готовый:
    # клиент /graph никогда не увидит «полузаписанный» PNG.
    temp = out / "monitoring.tmp.png"
    fig.savefig(temp)
    plt.close(fig)  # освобождаем память, иначе фигуры копятся

    # Готовый график заменяет предыдущий целиком.
    os.replace(temp, out / "monitoring.png")


async def sample_loop():
    """Раз в секунду записываем наблюдение и обновляем график."""
    out.mkdir(parents=True, exist_ok=True)

    # Один общий AsyncClient на весь цикл: переиспользует соединения.
    async with httpx.AsyncClient(timeout=3) as client:
        while True:
            row = {
                "time": datetime.now(UTC).isoformat(),
                "ok": False,
            }

            try:
                health = await client.get(model_url + "/health")
                metrics = await client.get(model_url + "/metrics")

                # raise_for_status — превращаем 4xx/5xx в исключение,
                # чтобы попасть в ветку except и записать ошибку.
                health.raise_for_status()
                metrics.raise_for_status()

                row.update(
                    {
                        "ok": True,
                        # metrics.json() отдаёт requests/5xx/predictions/
                        # warnings/active — всё это уходит в наблюдение.
                        **metrics.json(),
                    }
                )

            except (httpx.HTTPError, ValueError) as exc:
                # ValueError — на случай битого JSON в ответе.
                row["error"] = str(exc)

            # Критическая секция: счётчики и буфер samples меняются
            # только под lock, чтобы /metrics не прочитал полусостояние.
            with lock:
                state["polls"] += 1
                state["failures"] += int(not row["ok"])

                # Дублируем собственные счётчики монитора в строку —
                # так они попадают и в jsonl, и в график.
                row.update(
                    monitor_status="ok",
                    monitor_polls=state["polls"],
                    monitor_failures=state["failures"],
                )

                state["samples"].append(row)

                # Для графика храним последние 120 наблюдений.
                state["samples"] = state["samples"][-120:]
                # Копия — чтобы рендер шёл вне лока и не блокировал /metrics.
                samples = list(state["samples"])

            # Журнал наблюдений: append-only, удобно потом разбирать.
            with (out / "monitoring.jsonl").open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

            await asyncio.to_thread(render, samples)
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(_app):
    """Опрос запускается и завершается вместе с мониторингом."""
    task = asyncio.create_task(sample_loop())

    try:
        yield  # приложение работает, фоновый опрос идёт
    finally:
        # Корректная остановка: отменяем задачу и дожидаемся её,
        # проглатывая ожидаемый CancelledError.
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ЛР6: мониторинг",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    # Собственное здоровье монитора.
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with lock:
        return {
            "polls": state["polls"],
            "failures": state["failures"],
            # latest — последнее наблюдение; None, если ещё не опрашивали.
            "latest": (
                state["samples"][-1]
                if state["samples"]
                else None
            ),
        }


@app.get("/graph")
def graph():
    path = out / "monitoring.png"

    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="График ещё создаётся",
        )

    return FileResponse(
        path,
        media_type="image/png",
    )
