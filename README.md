# Фиолетовый агент для MLE-Bench

`MLE-Bench Purple Agent` — A2A-агент для автоматического решения табличных ML-задач в формате Kaggle/MLE-Bench.

Агент принимает архив `competition.tar.gz`, анализирует данные, подбирает стратегию обучения (в том числе с помощью LLM), обучает модель/ансамбль и возвращает `submission.csv`.

## Что делает агент

- Принимает входной архив с задачей (`train.csv`, `test.csv`, описание).
- Определяет тип задачи: `binary_classification`, `multiclass_classification` или `regression`.
- Выполняет препроцессинг табличных признаков (пропуски, категории, масштабирование, базовый feature engineering).
- Получает рекомендации по моделям через LLM (или использует эвристический fallback).
- Запускает кросс-валидацию, выбирает лучшую модель или обучает ансамбль.
- Формирует и возвращает `submission.csv`.

## Внешний интерфейс (A2A)

Агент публикует один основной скилл:

- `**mle-bench-ml-solver**` (`ML Competition Solver`)

Поддерживаемые режимы обмена:

- **Вход:** `application/gzip`, `text`
- **Выход:** `text/csv`, `text`
- **Streaming:** включен

## Внутренние инструменты (компоненты пайплайна)

Реализация в `src/purple_agent.py` состоит из следующих модулей:

- `**TaskAnalyzer`**
  - ищет train/test-файлы и описание задачи;
  - определяет `target` и `id` колонку;
  - вычисляет статистики датасета (размер, пропуски, типы признаков);
  - инферит тип ML-задачи.
- `**DataPreprocessor**`
  - безопасно выравнивает train/test по колонкам;
  - обрабатывает пропуски (медиана/мода);
  - кодирует категориальные признаки;
  - строит простые инженерные признаки;
  - применяет `RobustScaler`.
- `**LLMDecisionMaker**`
  - формирует структурированный prompt о датасете;
  - запрашивает рекомендации по моделям/препроцессингу;
  - валидирует и нормализует JSON-ответ;
  - имеет fallback на эвристики при недоступности LLM.
- `**ModelTrainer**`
  - оценивает кандидатов через cross-validation;
  - выбирает лучшую модель;
  - умеет собирать ансамбль (`stacking`, fallback в `voting`);
  - генерирует предсказания для submission.

## Какие модели использует агент

### Классификация

- `gradient_boosting` (`GradientBoostingClassifier`)
- `random_forest` (`RandomForestClassifier`)
- `extra_trees` (`ExtraTreesClassifier`)
- `logistic_regression` (`LogisticRegression`)
- `svm` (`SVC`)
- `knn` (`KNeighborsClassifier`)

Опционально (если библиотека установлена):

- `lightgbm` (`LGBMClassifier`)
- `xgboost` (`XGBClassifier`)
- `catboost` (`CatBoostClassifier`)

### Регрессия

- `gradient_boosting_regressor` (`GradientBoostingRegressor`)
- `random_forest_regressor` (`RandomForestRegressor`)
- `extra_trees_regressor` (`ExtraTreesRegressor`)
- `ridge` (`Ridge`)
- `lasso` (`Lasso`)
- `elastic_net` (`ElasticNet`)
- `svr` (`SVR`)
- `knn_regressor` (`KNeighborsRegressor`)

Опционально (если библиотека установлена):

- `lightgbm` / `lightgbm_regressor`
- `xgboost` (`XGBRegressor`)
- `catboost` (`CatBoostRegressor`)

## LLM и выбор стратегии

- Для LLM-рекомендаций используется OpenAI API (через `LLMDecisionMaker`).
- Модель можно передать через параметр `--openai-model`.
- Внутренний дефолт в логике рекомендаций — `gpt-5.4`.
- Если LLM недоступна, агент автоматически переходит на эвристический подбор моделей.

## Архитектура проекта

```text
server.py -> executor.py -> agent.py -> purple_agent.py
```

- `src/server.py` — поднимает A2A сервер и публикует agent card.
- `src/executor.py` — обработка A2A-задач и жизненного цикла task.
- `src/agent.py` — A2A-обертка: принимает архив и возвращает артефакт.
- `src/purple_agent.py` — ML-пайплайн (анализ, preprocessing, обучение, submission).

## Запуск локально

```bash
# Установить зависимости
uv sync

# Запустить агент
uv run src/server.py --host 0.0.0.0 --port 9009
```

С параметром модели:

```bash
uv run src/server.py --openai-model gpt-5.4
```

## Запуск в Docker

```bash
docker build -t purple-agent .
docker run -p 9009:9009 purple-agent
```

## Тесты

```bash
# Установить зависимости для тестов
uv sync --extra test

# Запуск тестов
uv run pytest --agent-url http://localhost:9009
```

## Переменные окружения

Основные переменные (см. `amber/amber-manifest-green.json5`):

- `OPENAI_API_KEY` — ключ OpenAI (опционально, если нужен LLM).
- `KAGGLE_USERNAME` — логин Kaggle.
- `KAGGLE_KEY` — API-ключ Kaggle.

## Ограничения

- Агент ориентирован на **табличные** задачи.
- Основной формат входных данных — CSV в архиве `competition.tar.gz`.
- Качество решения зависит от структуры данных, доступных библиотек и выбранной LLM-модели.

