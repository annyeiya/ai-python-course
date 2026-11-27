# ЛР6. FastAPI-сервис модели с переключением

Начало запуска UTC: 2026-11-27T19:26:17.043056+00:00

Коммит кода: aba17b52b9ddeac55e7c5f2db61fd28e76b53a0f
Код завершения pytest: 0; полнота и успешность прогона: True.

Модели: ml-v1 — ЛР2, dl-v1 — ЛР3. Метрики и хеши находятся в registry.json.

Использован один worker. При регистрации хеши вычисляются по текущим файлам моделей.

## Адреса и схемы

| Сервис | Метод и адрес | Назначение |
|---|---|---|
| модель | GET /health | состояние и активная версия |
| модель | POST /predict | request_id и 20 чисел → класс, вероятность, версия и предупреждение |
| модель | GET /metrics | запросы, 5xx, прогнозы и предупреждения |
| модель | GET /models | реестр версий и история |
| модель | POST /promote | продвижение версии |
| модель | POST /rollback | возврат предыдущей версии |
| монитор | GET /health | состояние монитора |
| монитор | GET /metrics | опросы и ошибки |
| монитор | GET /graph | график PNG |

Схема: ровно 20 конечных чисел в диапазоне [-1000, 1000]; ошибочные запросы получают 422.

## Хеш

HTTP 409; до: dl-v1; после: dl-v1; объяснение: {"detail":"SHA-256 модели не совпадает с реестром"}; проверка: True.

## Контракт

| Случай | Ожидается | Получено | Объяснение | Проверка |
|---|---:|---:|---|---|
| правильный вход | 200 | 200 | None | True |
| пропущено features | 422 | 422 | [{"type": "missing", "loc": ["body", "features"], "msg": "Field required", "input": {"request_id": "missing"}}] | True |
| неверный тип | 422 | 422 | [{"type": "float_type", "loc": ["body", "features", 0], "msg": "Input should be a valid number", "input": "wrong"}] | True |
| вне диапазона | 422 | 422 | [{"type": "less_than_equal", "loc": ["body", "features", 0], "msg": "Input should be less than or equal to 1000", "input": 1001, "ctx": {"le": 1000.0}}] | True |

## Нагрузка

| Адрес | Отправлено | Успешно | Ошибок | Примеры ошибок |
|---|---:|---:|---:|---|
| GET model/health | 100 | 100 | 0 | [] |
| GET model/metrics | 100 | 100 | 0 | [] |
| GET model/models | 100 | 100 | 0 | [] |
| POST model/predict | 100 | 100 | 0 | [] |
| POST model/promote | 100 | 100 | 0 | [] |
| POST model/rollback | 100 | 100 | 0 | [] |
| GET monitor/health | 100 | 100 | 0 | [] |
| GET monitor/metrics | 100 | 100 | 0 | [] |
| GET monitor/graph | 100 | 100 | 0 | [] |

## Примеры запросов и ответов обоих сервисов

Примеры — реальные ответы во время тестов; реестр и счётчики могли измениться позже. Для PNG показаны метаданные полученного ответа.

### GET model/health

**Запрос**

```http
GET http://127.0.0.1:8001/health
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "status": "ok",
  "active": "ml-v1"
}
```

### GET model/metrics

**Запрос**

```http
GET http://127.0.0.1:8001/metrics
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "requests": 115,
  "http_5xx": 0,
  "predictions": 2,
  "warnings": 1,
  "active": "ml-v1"
}
```

### GET model/models

**Запрос**

```http
GET http://127.0.0.1:8001/models
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "active": "ml-v1",
  "previous": "ml-v1",
  "versions": {
    "ml-v1": {
      "kind": "sklearn",
      "path": "reports/LAB2/ml_model.joblib",
      "sha256": "dcaab7b63186adede4c04309dde21f4674b00d51ba09befe925cfab7eb2b5314",
      "metrics": {
        "accuracy": 0.798,
        "pr_auc": 0.864095,
        "roc_auc": 0.874766
      },
      "status": "рабочая",
      "history": [
        {
          "at": "2026-11-27T17:56:13.538326+00:00",
          "status": "готовится",
          "reason": "регистрация артефакта модели"
        },
        {
          "at": "2026-11-27T17:56:13.538326+00:00",
          "status": "пробная",
          "reason": "хеш зафиксирован в реестре, метрики прочитаны из CSV"
        },
        {
          "at": "2026-11-27T17:56:13.538326+00:00",
          "status": "рабочая",
          "reason": "назначена начальной рабочей версией"
        },
        {
          "at": "2026-11-27T17:58:25.944941+00:00",
          "status": "снята",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T17:58:26.464850+00:00",
          "status": "рабочая",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T17:58:28.048205+00:00",
          "status": "снята",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T17:58:28.174566+00:00",
          "status": "рабочая",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T18:30:56.276118+00:00",
          "status": "снята",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T18:30:56.810557+00:00",
          "status": "рабочая",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T18:30:58.304016+00:00",
          "status": "снята",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T18:30:58.453795+00:00",
          "status": "рабочая",
          "reason": "откат на предыдущую версию"
        }
      ]
    },
    "dl-v1": {
      "kind": "torch",
      "path": "reports/LAB3/dl_model.pt",
      "sha256": "3cb6279e815e5e7b4f4b7d1ceb9a3c246d2fc08874842f335d2d7a60082f95b5",
      "metrics": {
        "accuracy": 0.989,
        "pr_auc": 0.991243,
        "roc_auc": 0.994114
      },
      "status": "снята",
      "history": [
        {
          "at": "2026-11-27T17:56:13.538326+00:00",
          "status": "готовится",
          "reason": "регистрация артефакта модели"
        },
        {
          "at": "2026-11-27T17:56:13.538326+00:00",
          "status": "пробная",
          "reason": "хеш зафиксирован в реестре, метрики прочитаны из CSV"
        },
        {
          "at": "2026-11-27T17:58:25.944941+00:00",
          "status": "рабочая",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T17:58:26.464850+00:00",
          "status": "снята",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T17:58:28.048205+00:00",
          "status": "рабочая",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T17:58:28.174566+00:00",
          "status": "снята",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T18:30:56.276118+00:00",
          "status": "рабочая",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T18:30:56.810557+00:00",
          "status": "снята",
          "reason": "откат на предыдущую версию"
        },
        {
          "at": "2026-11-27T18:30:58.304016+00:00",
          "status": "рабочая",
          "reason": "продвижение через API"
        },
        {
          "at": "2026-11-27T18:30:58.453795+00:00",
          "status": "снята",
          "reason": "откат на предыдущую версию"
        }
      ]
    }
  }
}
```

### POST model/predict

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "load-0",
  "features": [
    -2.2456075812,
    -2.3280841727,
    1.8478441751,
    0.8669442924,
    -0.3690202679,
    1.5179068801,
    -2.2456075812,
    5.4005776478,
    1.5179068801,
    -7.4109486412,
    -2.3280841727,
    -2.5518288108,
    1.9938237473,
    -1.7089053011,
    1.0413310887,
    1.5179068801,
    1.8307419463,
    1.7441509975,
    -2.3280841727,
    1.3869560646
  ]
}
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "request_id": "load-0",
  "version": "ml-v1",
  "prediction": 0,
  "probability": 0.03610049158199134,
  "validation": {
    "suspicious": false,
    "score": 0.6500693549111869
  }
}
```

### POST model/promote

**Запрос**

```http
POST http://127.0.0.1:8001/promote
Content-Type: application/json

{
  "version": "dl-v1"
}
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "active": "dl-v1",
  "changed": true
}
```

### POST model/rollback

**Запрос**

```http
POST http://127.0.0.1:8001/rollback
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "active": "ml-v1",
  "changed": true
}
```

### GET monitor/health

**Запрос**

```http
GET http://127.0.0.1:8002/health
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "status": "ok"
}
```

### GET monitor/metrics

**Запрос**

```http
GET http://127.0.0.1:8002/metrics
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "polls": 6,
  "failures": 0,
  "latest": {
    "time": "2026-11-27T19:26:39.249985+00:00",
    "ok": true,
    "requests": 539,
    "http_5xx": 0,
    "predictions": 102,
    "warnings": 2,
    "active": "ml-v1",
    "monitor_status": "ok",
    "monitor_polls": 6,
    "monitor_failures": 0
  }
}
```

### GET monitor/graph

**Запрос**

```http
GET http://127.0.0.1:8002/graph
```

**Ответ**

```text
HTTP 200
Content-Type: image/png

{
  "bytes": 29111,
  "sha256": "772a55eeea6558bd9f7819e07d88c6547c50150c8f7783390952c44c05863265",
  "note": "бинарный PNG, тело не вставлено в отчёт"
}
```

### Контракт /predict: правильный вход

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "contract-valid",
  "features": [
    -2.2456075812,
    -2.3280841727,
    1.8478441751,
    0.8669442924,
    -0.3690202679,
    1.5179068801,
    -2.2456075812,
    5.4005776478,
    1.5179068801,
    -7.4109486412,
    -2.3280841727,
    -2.5518288108,
    1.9938237473,
    -1.7089053011,
    1.0413310887,
    1.5179068801,
    1.8307419463,
    1.7441509975,
    -2.3280841727,
    1.3869560646
  ]
}
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "request_id": "contract-valid",
  "version": "ml-v1",
  "prediction": 0,
  "probability": 0.03610049158199134,
  "validation": {
    "suspicious": false,
    "score": 0.6500693549111869
  }
}
```

### Контракт /predict: пропущено features

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "missing"
}
```

**Ответ**

```text
HTTP 422
Content-Type: application/json

{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "features"
      ],
      "msg": "Field required",
      "input": {
        "request_id": "missing"
      }
    }
  ]
}
```

### Контракт /predict: неверный тип

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "contract-valid",
  "features": [
    "wrong",
    -2.3280841727,
    1.8478441751,
    0.8669442924,
    -0.3690202679,
    1.5179068801,
    -2.2456075812,
    5.4005776478,
    1.5179068801,
    -7.4109486412,
    -2.3280841727,
    -2.5518288108,
    1.9938237473,
    -1.7089053011,
    1.0413310887,
    1.5179068801,
    1.8307419463,
    1.7441509975,
    -2.3280841727,
    1.3869560646
  ]
}
```

**Ответ**

```text
HTTP 422
Content-Type: application/json

{
  "detail": [
    {
      "type": "float_type",
      "loc": [
        "body",
        "features",
        0
      ],
      "msg": "Input should be a valid number",
      "input": "wrong"
    }
  ]
}
```

### Контракт /predict: вне диапазона

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "contract-valid",
  "features": [
    1001,
    -2.3280841727,
    1.8478441751,
    0.8669442924,
    -0.3690202679,
    1.5179068801,
    -2.2456075812,
    5.4005776478,
    1.5179068801,
    -7.4109486412,
    -2.3280841727,
    -2.5518288108,
    1.9938237473,
    -1.7089053011,
    1.0413310887,
    1.5179068801,
    1.8307419463,
    1.7441509975,
    -2.3280841727,
    1.3869560646
  ]
}
```

**Ответ**

```text
HTTP 422
Content-Type: application/json

{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": [
        "body",
        "features",
        0
      ],
      "msg": "Input should be less than or equal to 1000",
      "input": 1001,
      "ctx": {
        "le": 1000.0
      }
    }
  ]
}
```

### Контроль предупреждения /predict

**Запрос**

```http
POST http://127.0.0.1:8001/predict
Content-Type: application/json

{
  "request_id": "warning-control",
  "features": [
    1000.0,
    -2.3280841727,
    1.8478441751,
    0.8669442924,
    -0.3690202679,
    1.5179068801,
    -2.2456075812,
    5.4005776478,
    1.5179068801,
    -7.4109486412,
    -2.3280841727,
    -2.5518288108,
    1.9938237473,
    -1.7089053011,
    1.0413310887,
    1.5179068801,
    1.8307419463,
    1.7441509975,
    -2.3280841727,
    1.3869560646
  ]
}
```

**Ответ**

```text
HTTP 200
Content-Type: application/json

{
  "request_id": "warning-control",
  "version": "ml-v1",
  "prediction": 0,
  "probability": 0.03610049158199134,
  "validation": {
    "suspicious": true,
    "score": 66415.09384954009
  }
}
```

## Дрилл

Отправлено 1000, корректных ответов 1000, нет корректного ответа 0, доля успеха 1.000000.

Продвижение HTTP 200; откат HTTP 200; версии ответов: {'ml-v1': 980, 'dl-v1': 20}; итог: ml-v1; предупреждений: 9.

Запросов в полёте перед продвижением/откатом: 16/19; клиентских перекрытий: 36/39; проверка: True.

## Валидатор ЛР4

Валидатор использует DL ЛР3 независимо от активной прогнозной модели. Предупреждение не блокирует прогноз; метка в запросе не передаётся.

### Как считается score

`score = max(d / T, c / max(confidence, 1e-12))`

`d` — расстояние Махаланобиса от масштабированной строки до ближайшего центра класса; `T` — сохранённый порог расстояния, откалиброванный на validation.

`p` — вероятность класса 1, рассчитанная DL-моделью валидатора; `confidence = max(p, 1 - p)` — уверенность в более вероятном классе; `c` — сохранённый порог уверенности.

Пороги T и c загружаются из validator_params.npz. Score безразмерен, не является вероятностью ошибки или корректности и не проверяет метку.

`suspicious = (d > T) OR (confidence < c)`. При score больше 1 есть предупреждение, при score меньше 1 — нет. Равенство 1 само по себе не является предупреждением, так как сравнения строгие.

Контроль: HTTP 200; {"suspicious": true, "score": 66415.09384954009}; проверка: True.

## Мониторинг

Состояние: {"polls": 12, "failures": 0, "latest": {"time": "2026-11-27T19:26:47.265118+00:00", "ok": true, "requests": 1635, "http_5xx": 0, "predictions": 1102, "warnings": 11, "active": "ml-v1", "monitor_status": "ok", "monitor_polls": 12, "monitor_failures": 0}}; проверка: True.

## Результаты pytest

| Тест | Что проверяет | Результат | Причина при неуспехе |
|---|---|---|---|
| test_contract_schema | Pydantic проверяет число, тип и диапазон признаков. | Пройден (PASSED) |  |
| test_bad_hash_does_not_switch | Неверный хеш не меняет модель и реестр в модульной проверке. | Пройден (PASSED) |  |
| test_warning_does_not_block_prediction[False] | Предупреждение не удаляет прогноз из ответа; используются заглушки. | Пройден (PASSED) |  |
| test_warning_does_not_block_prediction[True] | Предупреждение не удаляет прогноз из ответа; используются заглушки. | Пройден (PASSED) |  |
| test_contract[правильный вход-200] | Проверяется HTTP-контракт /predict для указанного случая. | Пройден (PASSED) |  |
| test_contract[пропущено features-422] | Проверяется HTTP-контракт /predict для указанного случая. | Пройден (PASSED) |  |
| test_contract[неверный тип-422] | Проверяется HTTP-контракт /predict для указанного случая. | Пройден (PASSED) |  |
| test_contract[вне диапазона-422] | Проверяется HTTP-контракт /predict для указанного случая. | Пройден (PASSED) |  |
| test_bad_hash | Сервис отклоняет неверный хеш без смены активной версии. | Пройден (PASSED) |  |
| test_warning | Экстремальная строка получает предупреждение и прогноз HTTP 200. | Пройден (PASSED) |  |
| test_load_route[GET model/health] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[GET model/metrics] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[GET model/models] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[POST model/predict] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[POST model/promote] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[POST model/rollback] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[GET monitor/health] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[GET monitor/metrics] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_load_route[GET monitor/graph] | Сто HTTP-запросов к указанному адресу без ошибок. | Пройден (PASSED) |  |
| test_switch_drill | Проверяются прогнозы во время продвижения и отката модели. | Пройден (PASSED) |  |
| test_monitor | Проверяются опросы мониторинга и наличие графика. | Пройден (PASSED) |  |

## Выводы

Общий код pytest: 0; подробный журнал — service_tests.log.
Результат относится только к данному короткому прогону с одним worker. Параллельные управляющие команды, межпроцессная синхронизация, авторизация и TLS не проверяются.
Данные о перекрытии относятся к интервалам HTTP на стороне клиента. Повторные promote/rollback в нагрузочном тесте в основном идемпотентны.
