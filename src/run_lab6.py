"""ЛР6: запускаем pytest и создаём отчёт по результатам тестов."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("reports/LAB6")
EVENTS = OUT / "lab6_measurements.jsonl"
XML = OUT / "pytest_lab6.xml"


def cell(value):
    """Экранируем текст для таблицы Markdown."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def read_events():
    """Читаем наблюдения текущего прогона."""
    if not EVENTS.is_file():
        return []

    return [
        json.loads(line)
        for line in EVENTS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def read_cases():
    """Читаем результаты pytest из JUnit XML."""
    if not XML.is_file():
        return []

    cases = []

    for item in ET.parse(XML).iter("testcase"):
        issue = item.find("failure")
        status = "FAILED" if issue is not None else "PASSED"

        if issue is None:
            issue = item.find("error")

            if issue is not None:
                status = "ERROR"

        if issue is None:
            issue = item.find("skipped")

            if issue is not None:
                status = "SKIPPED"

        cases.append(
            {
                "name": item.get("name", ""),
                "status": status,
                "detail": (
                    ""
                    if issue is None
                    else (
                        issue.get("message")
                        or (issue.text or "")
                    )
                ),
            }
        )

    return cases


def complete(events, cases, code):
    """Не считаем прогон успешным, если проверки пропущены."""
    sections = (
        "contract",
        "hash",
        "warning",
        "load",
        "drill",
        "monitor",
    )

    counts = {
        section: sum(
            event["section"] == section
            for event in events
        )
        for section in sections
    }

    expected = {
        "contract": 4,
        "hash": 1,
        "warning": 1,
        "load": 9,
        "drill": 1,
        "monitor": 1,
    }

    return (
        code == 0
        and counts == expected
        and len(cases) >= 21
        and all(
            case["status"] == "PASSED"
            for case in cases
        )
    )

def add_example(lines, example):
    """Добавляем фактическую пару HTTP-запрос/ответ в отчёт."""
    if example is None:
        lines.append(
            "Пример не сохранился. См. журнал pytest."
        )
        return

    lines += [
        "**Запрос**",
        "",
        "```http",
        f"{example['method']} {example['url']}",
    ]

    if example.get("request_body") is not None:
        lines += [
            "Content-Type: application/json",
            "",
            json.dumps(
                example["request_body"],
                ensure_ascii=False,
                indent=2,
            ),
        ]

    lines += [
        "```",
        "",
        "**Ответ**",
        "",
        "```text",
    ]

    if "response" in example:
        response = example["response"]

        lines += [
            f"HTTP {response['status']}",
            f"Content-Type: {response['content_type']}",
            "",
            json.dumps(
                response["body"],
                ensure_ascii=False,
                indent=2,
            ),
        ]

    else:
        lines.append(
            "Ответ не получен: "
            f"{example.get('error', 'причина неизвестна')}"
        )

    lines.append("```")

# Кратко описываем назначение каждого теста для итоговой таблицы.
TEST_DESCRIPTIONS = {
    "test_contract_schema": (
        "Pydantic проверяет число, тип и диапазон признаков."
    ),
    "test_bad_hash_does_not_switch": (
        "Неверный хеш не меняет модель и реестр "
        "в модульной проверке."
    ),
    "test_warning_does_not_block_prediction": (
        "Предупреждение не удаляет прогноз из ответа; "
        "используются заглушки."
    ),
    "test_contract": (
        "Проверяется HTTP-контракт /predict "
        "для указанного случая."
    ),
    "test_bad_hash": (
        "Сервис отклоняет неверный хеш "
        "без смены активной версии."
    ),
    "test_warning": (
        "Экстремальная строка получает предупреждение "
        "и прогноз HTTP 200."
    ),
    "test_load_route": (
        "Сто HTTP-запросов к указанному адресу без ошибок."
    ),
    "test_switch_drill": (
        "Проверяются прогнозы во время продвижения "
        "и отката модели."
    ),
    "test_monitor": (
        "Проверяются опросы мониторинга и наличие графика."
    ),
}


def test_description(name):
    """Находим назначение теста по имени без параметров pytest."""
    base_name = name.split("[", 1)[0]

    return TEST_DESCRIPTIONS.get(
        base_name,
        "Описание для этого теста не задано.",
    )


def display_test_name(name):
    """Делаем кириллицу в параметрах pytest читаемой."""
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        name,
    )


def test_result(status):
    """Переводим статус pytest, не считая пропуск успехом."""
    return {
        "PASSED": "Пройден",
        "FAILED": "Не пройден",
        "ERROR": "Ошибка запуска",
        "SKIPPED": "Пропущен",
    }.get(status, "Неизвестно")


def write_report(
    events,
    cases,
    code,
    started,
    passed,
):
    """Формируем отчёт и при неуспешном прогоне."""
    groups = {}

    for event in events:
        groups.setdefault(
            event["section"],
            [],
        ).append(event)

    lines = [
        "# ЛР6. FastAPI-сервис модели с переключением",
        "",
        f"Начало запуска UTC: {started}",
        "",
        "Коммит кода: ",
        (
            f"Код завершения pytest: {code}; "
            f"полнота и успешность прогона: {passed}."
        ),
        "",
        (
            "Модели: ml-v1 — ЛР2, dl-v1 — ЛР3. "
            "Метрики и хеши находятся в registry.json."
        ),
        "",
        (
            "Использован один worker. При регистрации хеши "
            "вычисляются по текущим файлам моделей."
        ),
        "",
        "## Адреса и схемы",
        "",
        "| Сервис | Метод и адрес | Назначение |",
        "|---|---|---|",
        "| модель | GET /health | состояние и активная версия |",
        (
            "| модель | POST /predict | request_id и 20 чисел "
            "→ класс, вероятность, версия и предупреждение |"
        ),
        (
            "| модель | GET /metrics | запросы, 5xx, "
            "прогнозы и предупреждения |"
        ),
        "| модель | GET /models | реестр версий и история |",
        "| модель | POST /promote | продвижение версии |",
        "| модель | POST /rollback | возврат предыдущей версии |",
        "| монитор | GET /health | состояние монитора |",
        "| монитор | GET /metrics | опросы и ошибки |",
        "| монитор | GET /graph | график PNG |",
        "",
        (
            "Схема: ровно 20 конечных чисел в диапазоне "
            "[-1000, 1000]; ошибочные запросы получают 422."
        ),
        "",
        "## Хеш",
        "",
    ]

    if groups.get("hash"):
        for row in groups["hash"]:
            lines.append(
                f"HTTP {row['status']}; "
                f"до: {cell(row['before'])}; "
                f"после: {cell(row['after'])}; "
                f"объяснение: {cell(row['detail'])}; "
                f"проверка: {row['passed']}."
            )
    else:
        lines.append(
            "Нет результата: тест не дошёл до сохранения "
            "наблюдения. См. журнал pytest."
        )

    lines += [
        "",
        "## Контракт",
        "",
        "| Случай | Ожидается | Получено | Объяснение | Проверка |",
        "|---|---:|---:|---|---|",
    ]

    for row in groups.get("contract", []):
        lines.append(
            f"| {cell(row['name'])} | {row['expected']} | "
            f"{row['status']} | {cell(row['detail'])} | "
            f"{row['passed']} |"
        )

    if not groups.get("contract"):
        lines.append(
            "| Нет наблюдений | — | — | "
            "См. журнал pytest | не выполнено |"
        )

    lines += [
        "",
        "## Нагрузка",
        "",
        "| Адрес | Отправлено | Успешно | Ошибок | Примеры ошибок |",
        "|---|---:|---:|---:|---|",
    ]

    for row in groups.get("load", []):
        lines.append(
            f"| {cell(row['name'])} | {row['sent']} | "
            f"{row['ok']} | {row['errors']} | "
            f"{cell(row['examples'])} |"
        )

    if not groups.get("load"):
        lines.append(
            "| Нет наблюдений | — | — | — | См. журнал pytest |"
        )

    # Примеры взяты из HTTP-вызовов, которые выполнил pytest.
    lines += [
        "",
        "## Примеры запросов и ответов обоих сервисов",
        "",
        (
            "Примеры — реальные ответы во время тестов; "
            "реестр и счётчики могли измениться позже. "
            "Для PNG показаны метаданные полученного ответа."
        ),
    ]

    # По одному примеру для каждого адреса обоих сервисов.
    for row in groups.get("load", []):
        lines += [
            "",
            f"### {row['name']}",
            "",
        ]

        add_example(
            lines,
            row.get("example"),
        )

    if not groups.get("load"):
        lines += [
            "",
            (
                "Примеры отсутствуют: нагрузочные тесты "
                "не записали наблюдения."
            ),
        ]

    # Отдельно показываем все контрактные случаи /predict.
    for row in groups.get("contract", []):
        lines += [
            "",
            f"### Контракт /predict: {row['name']}",
            "",
        ]

        add_example(
            lines,
            row.get("example"),
        )

    # Показываем пример предупреждения валидатора.
    for row in groups.get("warning", []):
        lines += [
            "",
            "### Контроль предупреждения /predict",
            "",
        ]

        add_example(
            lines,
            row.get("example"),
        )

    lines += [
        "",
        "## Дрилл",
        "",
    ]

    if groups.get("drill"):
        row = groups["drill"][-1]

        lines.extend(
            [
                (
                    f"Отправлено {row['sent']}, "
                    f"корректных ответов {row['ok']}, "
                    f"нет корректного ответа {row['lost']}, "
                    f"доля успеха {row['ratio']:.6f}."
                ),
                "",
                (
                    f"Продвижение HTTP {row['promote']}; "
                    f"откат HTTP {row['rollback']}; "
                    f"версии ответов: {row['versions']}; "
                    f"итог: {row['final']}; "
                    f"предупреждений: {row['warnings']}."
                ),
                "",
                (
                    "Запросов в полёте перед продвижением/откатом: "
                    f"{row['inflight_before_promote']}/"
                    f"{row['inflight_before_rollback']}; "
                    "клиентских перекрытий: "
                    f"{row['overlap_promote']}/"
                    f"{row['overlap_rollback']}; "
                    f"проверка: {row['passed']}."
                ),
            ]
        )

    else:
        lines.append(
            "Нет результата дрилла. См. журнал pytest."
        )

    lines += [
        "",
        "## Валидатор ЛР4",
        "",
        (
            "Валидатор использует DL ЛР3 независимо от активной "
            "прогнозной модели. Предупреждение не блокирует прогноз; "
            "метка в запросе не передаётся."
        ),
        "",
        "### Как считается score",
        "",
        "`score = max(d / T, c / max(confidence, 1e-12))`",
        "",
        (
            "`d` — расстояние Махаланобиса от масштабированной "
            "строки до ближайшего центра класса; `T` — сохранённый "
            "порог расстояния, откалиброванный на validation."
        ),
        "",
        (
            "`p` — вероятность класса 1, рассчитанная DL-моделью "
            "валидатора; `confidence = max(p, 1 - p)` — уверенность "
            "в более вероятном классе; `c` — сохранённый порог "
            "уверенности."
        ),
        "",
        (
            "Пороги T и c загружаются из validator_params.npz. "
            "Score безразмерен, не является вероятностью ошибки "
            "или корректности и не проверяет метку."
        ),
        "",
        (
            "`suspicious = (d > T) OR (confidence < c)`. "
            "При score больше 1 есть предупреждение, при score "
            "меньше 1 — нет. Равенство 1 само по себе не является "
            "предупреждением, так как сравнения строгие."
        ),
        "",
    ]

    for row in groups.get("warning", []):
        lines.append(
            f"Контроль: HTTP {row['status']}; "
            f"{cell(row['validation'])}; "
            f"проверка: {row['passed']}."
        )

    if not groups.get("warning"):
        lines.append(
            "Нет наблюдения контроля предупреждения. "
            "См. журнал pytest."
        )

    lines += [
        "",
        "## Мониторинг",
        "",
    ]

    for row in groups.get("monitor", []):
        lines.append(
            f"Состояние: {cell(row['data'])}; "
            f"проверка: {row['passed']}."
        )

    if not groups.get("monitor"):
        lines.append(
            "Нет наблюдения мониторинга. См. журнал pytest."
        )

    lines += [
        "",
        "## Результаты pytest",
        "",
        (
            "| Тест | Что проверяет | Результат | "
            "Причина при неуспехе |"
        ),
        "|---|---|---|---|",
    ]

    for row in cases:
        # Имя и статус берём из результата реального запуска pytest.
        name = display_test_name(row["name"])
        description = test_description(row["name"])

        result = (
            f"{test_result(row['status'])} "
            f"({row['status']})"
        )

        lines.append(
            f"| {cell(name)} | {cell(description)} | "
            f"{cell(result)} | {cell(row['detail'])} |"
        )

    if not cases:
        lines.append(
            "| Нет результатов | "
            "Не удалось получить результаты pytest | "
            "Ошибка | См. service_tests.log |"
        )

    lines += [
        "",
        "## Выводы",
        "",
        (
            f"Общий код pytest: {code}; "
            "подробный журнал — service_tests.log."
        ),
        (
            "Результат относится только к данному короткому "
            "прогону с одним worker. Параллельные управляющие "
            "команды, межпроцессная синхронизация, авторизация "
            "и TLS не проверяются."
        ),
        (
            "Данные о перекрытии относятся к интервалам HTTP "
            "на стороне клиента. Повторные promote/rollback "
            "в нагрузочном тесте в основном идемпотентны."
        ),
        "",
    ]

    (OUT / "lab6_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Не используем измерения и XML прошлого запуска.
    for path in (EVENTS, XML):
        path.unlink(missing_ok=True)

    started = datetime.now(UTC).isoformat()

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_lab6_service.py",
        "tests/test_lab6_http.py",
        "-v",
        "--tb=short",
        "--junitxml",
        str(XML),
    ]

    print(
        "[pipeline] Запускаем pytest: " + " ".join(command),
        flush=True,
    )

    # Включаем HTTP-тесты для полного прогона ЛР6.
    env = {
        **os.environ,
        "LAB6_HTTP_TESTS": "1",
    }

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    events = read_events()
    cases = read_cases()

    passed = complete(
        events,
        cases,
        result.returncode,
    )

    log = [
        "ЛР6: результаты pytest и наблюдения одного прогона",
        f"Начало UTC: {started}",
        f"Код pytest: {result.returncode}",
        f"Полный прогон успешен: {passed}",
        "",
        "[вывод pytest]",
        result.stdout,
        result.stderr,
        "[наблюдения тестов]",
    ]

    log.extend(
        json.dumps(row, ensure_ascii=False)
        for row in events
    )

    (OUT / "service_tests.log").write_text(
        "\n".join(log) + "\n",
        encoding="utf-8",
    )

    write_report(
        events,
        cases,
        result.returncode,
        started,
        passed,
    )

    print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    print(
        "[pipeline] Отчёт и журнал сохранены в reports/LAB6/"
    )

    return 0 if passed else (result.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
