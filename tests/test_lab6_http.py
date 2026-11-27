"""ЛР6: интеграционные HTTP-тесты и измерения для отчёта."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd
import pytest

OUT = Path("reports/LAB6")
MODEL = "http://127.0.0.1:8001"
MONITOR = "http://127.0.0.1:8002"
EVENTS = OUT / "lab6_measurements.jsonl"

COUNT = 100
DRILL = 1000

# CI без артефактов не запускает HTTP-прогон автоматически.
pytestmark = pytest.mark.skipif(
    os.environ.get("LAB6_HTTP_TESTS") != "1",
    reason="HTTP-прогон включается через LAB6_HTTP_TESTS=1",
)


def save_event(section, **data):
    """Сохраняем наблюдение до assert, чтобы видеть и провалы."""
    with EVENTS.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"section": section, **data},
                ensure_ascii=False,
            )
            + "\n"
        )


def free_port(port):
    """Проверяем, не занят ли порт."""
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start(module, port, env, log):
    """Создаём отдельный процесс сервиса и его журнал."""
    with (OUT / log).open("w", encoding="utf-8") as file:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                module + ":app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            env=env,
            stdout=file,
            stderr=subprocess.STDOUT,
        )


def stop(process):
    """Завершаем только процесс, запущенный тестами."""
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_ready(url, process):
    """Проверяем настоящий ответ сервиса и его состояние."""
    end = time.monotonic() + 45

    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(
                f"Сервис {url} завершился: проверьте server log"
            )

        try:
            response = httpx.get(
                url + "/health",
                timeout=1,
            )

            if response.status_code == 200:
                return

        except httpx.HTTPError:
            pass

        time.sleep(0.2)

    raise RuntimeError(
        f"Сервис {url} не запустился за 45 секунд"
    )


def valid_warning(body):
    """Проверяем поля предупреждения валидатора."""
    warning = body.get("validation")

    return (
        isinstance(warning, dict)
        and type(warning.get("suspicious")) is bool
        and type(warning.get("score")) in (int, float)
        and math.isfinite(warning["score"])
    )


def valid_prediction(body, request_id):
    """Проверяем содержимое успешного ответа прогноза."""
    probability = body.get("probability")

    return (
        body.get("request_id") == request_id
        and body.get("version") in ("ml-v1", "dl-v1")
        and type(probability) in (int, float)
        and math.isfinite(probability)
        and 0 <= probability <= 1
        and type(body.get("prediction")) is int
        and body["prediction"] in (0, 1)
        and valid_warning(body)
    )

def response_example(response):
    """Сохраняем HTTP-ответ; PNG описываем метаданными."""
    content_type = response.headers.get("content-type", "")

    if content_type.lower().startswith("image/png"):
        body = {
            "bytes": len(response.content),
            "sha256": hashlib.sha256(
                response.content
            ).hexdigest(),
            "note": "бинарный PNG, тело не вставлено в отчёт",
        }

    else:
        try:
            body = response.json()
        except ValueError:
            # Если ответ не JSON, сохраняем его текст.
            body = response.text

    return {
        "status": response.status_code,
        "content_type": content_type,
        "body": body,
    }


@pytest.fixture(scope="session")
def lab6():
    """Запускаем сервисы один раз и завершаем их после тестов."""
    OUT.mkdir(parents=True, exist_ok=True)

    # Не смешиваем наблюдения разных запусков.
    EVENTS.write_text("", encoding="utf-8")

    for name in ("monitoring.jsonl", "monitoring.png"):
        (OUT / name).unlink(missing_ok=True)

    if not all(
        free_port(port)
        for port in (8001, 8002, 8003)
    ):
        raise RuntimeError(
            "Порты 8001–8003 должны быть свободны"
        )

    if not (OUT / "registry.json").is_file():
        raise RuntimeError(
            "Сначала выполните python src/prepare_registry.py"
        )

    cols = [f"feature_{i:02d}" for i in range(20)]

    rows = (
        pd.read_csv("data/generated/LAB5/similar.csv")[cols]
        .head(DRILL)
        .astype(float)
        .values
        .tolist()
    )

    if not rows or any(
        not math.isfinite(value) or abs(value) > 1000
        for row in rows
        for value in row
    ):
        raise ValueError(
            "Строки ЛР5 отсутствуют либо выходят за диапазон API"
        )

    env = {
        **os.environ,
        "PYTHONPATH": str(Path("src").resolve()),
        "LAB6_REGISTRY": str(
            (OUT / "registry.json").resolve()
        ),
        "LAB6_MONITOR_DIR": str(OUT.resolve()),
        "LAB6_MODEL_URL": MODEL,
    }

    processes = []

    try:
        model = start(
            "model_service",
            8001,
            env,
            "model_server.log",
        )

        processes.append(model)
        wait_ready(MODEL, model)

        monitor = start(
            "monitor_service",
            8002,
            env,
            "monitor_server.log",
        )

        processes.append(monitor)
        wait_ready(MONITOR, monitor)

        with httpx.Client(timeout=20) as client:
            yield {
                "client": client,
                "rows": rows,
                "env": env,
            }

    finally:
        for process in reversed(processes):
            stop(process)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("правильный вход", 200),
        ("пропущено features", 422),
        ("неверный тип", 422),
        ("вне диапазона", 422),
    ],
)
def test_contract(lab6, case, expected):
    """Шаги 1 и 5: HTTP-ответ и объяснение ошибки 422."""
    features = lab6["rows"][0]

    body = {
        "request_id": "contract-valid",
        "features": features,
    }

    if case == "пропущено features":
        body = {"request_id": "missing"}

    elif case == "неверный тип":
        body["features"] = ["wrong", *features[1:]]

    elif case == "вне диапазона":
        body["features"] = [1001, *features[1:]]

    response = lab6["client"].post(
        MODEL + "/predict",
        json=body,
    )

    data = response.json()

    detail = (
        data.get("detail")
        if expected == 422
        else None
    )

    passed = response.status_code == expected and (
        bool(detail)
        if expected == 422
        else valid_prediction(data, "contract-valid")
    )

    save_event(
        "contract",
        name=case,
        expected=expected,
        status=response.status_code,
        detail=detail,
        passed=passed,
        example={
            "method": "POST",
            "url": str(response.request.url),
            "request_body": body,
            "response": response_example(response),
        },
    )

    assert passed, (
        f"Контракт {case}: HTTP {response.status_code}, "
        f"ответ {data}"
    )


def test_bad_hash(lab6):
    """Шаг 3: неверный хеш ML не меняет активную DL."""
    client = lab6["client"]
    data = client.get(MODEL + "/models").json()

    # Изменяем только временную копию реестра.
    data["active"] = "dl-v1"
    data["versions"]["dl-v1"]["status"] = "рабочая"
    data["versions"]["ml-v1"]["status"] = "пробная"
    data["versions"]["ml-v1"]["sha256"] = "0" * 64

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "registry.json"

        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

        server = start(
            "model_service",
            8003,
            {
                **lab6["env"],
                "LAB6_REGISTRY": str(path),
            },
            "bad_hash_server.log",
        )

        try:
            url = "http://127.0.0.1:8003"
            wait_ready(url, server)

            before = client.get(
                url + "/health"
            ).json()["active"]

            response = client.post(
                url + "/promote",
                json={"version": "ml-v1"},
            )

            after = client.get(
                url + "/health"
            ).json()["active"]

            passed = (
                response.status_code == 409
                and before == after
                and "SHA-256" in response.text
            )

            save_event(
                "hash",
                status=response.status_code,
                before=before,
                after=after,
                detail=response.text,
                passed=passed,
            )

            assert passed

        finally:
            stop(server)


def test_warning(lab6):
    """Экстремальная строка получает прогноз и предупреждение."""
    features = [
        1000.0,
        *lab6["rows"][0][1:],
    ]

    response = lab6["client"].post(
        MODEL + "/predict",
        json={
            "request_id": "warning-control",
            "features": features,
        },
    )

    body = response.json()

    passed = (
        response.status_code == 200
        and valid_prediction(body, "warning-control")
        and body["validation"]["suspicious"] is True
    )

    save_event(
        "warning",
        status=response.status_code,
        validation=body.get("validation"),
        passed=passed,
        example={
            "method": "POST",
            "url": str(response.request.url),
            "request_body": {
                "request_id": "warning-control",
                "features": features,
            },
            "response": response_example(response),
        },
    )

    assert passed, (
        f"Предупреждение не подтверждено: {body}"
    )


ROUTES = [
    ("GET model/health", "GET", MODEL + "/health"),
    ("GET model/metrics", "GET", MODEL + "/metrics"),
    ("GET model/models", "GET", MODEL + "/models"),
    ("POST model/predict", "POST", MODEL + "/predict"),
    ("POST model/promote", "POST", MODEL + "/promote"),
    ("POST model/rollback", "POST", MODEL + "/rollback"),
    ("GET monitor/health", "GET", MONITOR + "/health"),
    ("GET monitor/metrics", "GET", MONITOR + "/metrics"),
    ("GET monitor/graph", "GET", MONITOR + "/graph"),
]


@pytest.mark.parametrize(
    "name,method,url",
    ROUTES,
    ids=[row[0] for row in ROUTES],
)
def test_load_route(lab6, name, method, url):
    """Шаг 5: по 100 HTTP-вызовов каждого адреса."""
    client = lab6["client"]
    rows = lab6["rows"]

    # Перед нагрузкой на rollback нужна предыдущая версия.
    if url.endswith("/rollback"):
        client.post(
            MODEL + "/promote",
            json={"version": "dl-v1"},
        ).raise_for_status()

    if url.endswith("/graph"):
        deadline = time.monotonic() + 30

        while (
            time.monotonic() < deadline
            and client.get(url).status_code != 200
        ):
            time.sleep(0.2)

    ok = 0
    errors = []

    # Сохраним первый запрос и ответ для отчёта.
    example = None

    for i in range(COUNT):
        if url.endswith("/predict"):
            payload = {
                "request_id": f"load-{i}",
                "features": rows[i % len(rows)],
            }

        elif url.endswith("/promote"):
            payload = {"version": "dl-v1"}

        else:
            payload = None

        try:
            response = client.request(
                method,
                url,
                json=payload,
            )

            # Берём пример из реального нагрузочного теста.
            if i == 0:
                example = {
                    "method": method,
                    "url": str(response.request.url),
                    "request_body": payload,
                    "response": response_example(response),
                }

            valid = response.status_code == 200

            if url.endswith("/predict") and valid:
                valid = valid_prediction(
                    response.json(),
                    f"load-{i}",
                )

            ok += int(valid)

            if not valid and len(errors) < 3:
                errors.append(
                    f"запрос {i}: HTTP {response.status_code}"
                )

        except (httpx.HTTPError, ValueError) as exc:
            # Если первый запрос не получил ответ, сохраняем ошибку.
            if i == 0 and example is None:
                example = {
                    "method": method,
                    "url": url,
                    "request_body": payload,
                    "error": str(exc),
                }

            if len(errors) < 3:
                errors.append(f"запрос {i}: {exc}")

    save_event(
        "load",
        name=name,
        sent=COUNT,
        ok=ok,
        errors=COUNT - ok,
        examples=errors,
        example=example,
    )

    assert ok == COUNT, (
        f"{name}: успешно {ok}/{COUNT}, примеры: {errors}"
    )


def test_switch_drill(lab6):
    """Шаг 4: прогнозы во время продвижения и отката."""
    client = lab6["client"]
    rows = lab6["rows"]

    client.post(
        MODEL + "/promote",
        json={"version": "ml-v1"},
    ).raise_for_status()

    lock = threading.Lock()
    current = {"inflight": 0}
    records = []

    def request_one(i):
        """Один запрос прогноза с уникальным идентификатором."""
        time.sleep(0.015)

        request_id = f"drill-{i}"
        started = time.monotonic()

        with lock:
            current["inflight"] += 1

        try:
            response = client.post(
                MODEL + "/predict",
                json={
                    "request_id": request_id,
                    "features": rows[i % len(rows)],
                },
            )

            body = (
                response.json()
                if response.status_code == 200
                else {}
            )

            ok = (
                response.status_code == 200
                and valid_prediction(body, request_id)
            )

            record = {
                "id": request_id,
                "ok": ok,
                "version": body.get("version"),
                "warning": body.get(
                    "validation",
                    {},
                ).get("suspicious", False),
            }

        except (httpx.HTTPError, ValueError) as exc:
            record = {
                "id": request_id,
                "ok": False,
                "version": None,
                "error": str(exc),
            }

        finally:
            with lock:
                current["inflight"] -= 1

        record.update(
            start=started,
            end=time.monotonic(),
        )

        with lock:
            records.append(record)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=20
    ) as executor:
        futures = [
            executor.submit(request_one, i)
            for i in range(DRILL)
        ]

        deadline = time.monotonic() + 60

        while time.monotonic() < deadline:
            with lock:
                ready = (
                    len(records) >= 5
                    and current["inflight"] > 0
                )

            if ready:
                break

            time.sleep(0.005)

        with lock:
            before_promote = current["inflight"]

        promote_start = time.monotonic()

        promoted = client.post(
            MODEL + "/promote",
            json={"version": "dl-v1"},
        )

        promote_end = time.monotonic()

        while time.monotonic() < deadline:
            with lock:
                ready = (
                    any(
                        row["ok"]
                        and row["version"] == "dl-v1"
                        for row in records
                    )
                    and current["inflight"] > 0
                )

            if ready:
                break

            time.sleep(0.005)

        with lock:
            before_rollback = current["inflight"]

        rollback_start = time.monotonic()

        rolled = client.post(
            MODEL + "/rollback"
        )

        rollback_end = time.monotonic()

        for future in futures:
            future.result()

    ids = [
        row["id"]
        for row in records
        if row["ok"]
    ]

    success = len(set(ids))

    versions = dict(
        Counter(
            row["version"]
            for row in records
            if row["ok"]
        )
    )

    final = client.get(
        MODEL + "/health"
    ).json()["active"]

    overlap_promote = sum(
        row["start"] <= promote_end
        and row["end"] >= promote_start
        for row in records
    )

    overlap_rollback = sum(
        row["start"] <= rollback_end
        and row["end"] >= rollback_start
        for row in records
    )

    passed = (
        success == DRILL
        and len(ids) == DRILL
        and before_promote > 0
        and before_rollback > 0
        and overlap_promote > 0
        and overlap_rollback > 0
        and promoted.status_code == 200
        and promoted.json().get("active") == "dl-v1"
        and promoted.json().get("changed") is True
        and rolled.status_code == 200
        and rolled.json().get("active") == "ml-v1"
        and rolled.json().get("changed") is True
        and final == "ml-v1"
        and versions.get("ml-v1", 0) > 0
        and versions.get("dl-v1", 0) > 0
    )

    result = {
        "sent": DRILL,
        "ok": success,
        "lost": DRILL - success,
        "ratio": success / DRILL,
        "promote": promoted.status_code,
        "rollback": rolled.status_code,
        "inflight_before_promote": before_promote,
        "inflight_before_rollback": before_rollback,
        "overlap_promote": overlap_promote,
        "overlap_rollback": overlap_rollback,
        "versions": versions,
        "final": final,
        "passed": passed,
        "warnings": sum(
            bool(row.get("warning"))
            for row in records
            if row["ok"]
        ),
    }

    save_event("drill", **result)

    assert passed, (
        f"Дрилл не выполнен: {result}"
    )


def test_monitor(lab6):
    """Шаг 6: проверяем опросы мониторинга и график."""
    time.sleep(2)

    data = lab6["client"].get(
        MONITOR + "/metrics"
    ).json()

    passed = (
        data["polls"] > 0
        and data["failures"] == 0
        and bool(data["latest"])
        and data["latest"]["ok"]
        and (OUT / "monitoring.png").is_file()
    )

    save_event(
        "monitor",
        data=data,
        passed=passed,
    )

    assert passed, f"Мониторинг: {data}"
