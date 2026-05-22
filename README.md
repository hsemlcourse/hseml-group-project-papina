[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Предсказание направления изменения ключевой ставки

**Студент:** Папина Анжелика Владимировна

**Группа:** БИВ235

## Задача

Классификация направления изменения ключевой ставки центрального банка: `Hike`, `Hold`, `Cut`.

**Целевая метрика:** Macro F1-score.

## Структура репозитория

```
.
├── data
│   ├── processed/processed_data.csv
│   └── raw/global_central_bank_rates_1945_2026.csv
├── models
│   └── final_model.pkl
├── notebooks
│   └── cp2.ipynb
├── report
│   └── report.md
├── screenshots
│   ├── swagger_ui.png
│   ├── swagger_post_request.png
│   └── swagger_response.png
├── src
│   ├── __init__.py
│   ├── main.py
│   └── model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .flake8
└── README.md
```

## Быстрый старт

### 1. Клонирование репозитория

```
git clone <repo-url>
cd <repo-folder>
```

### 2. Установка зависимостей

```
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Запуск Jupyter Notebook

```
jupyter notebook notebooks/cp2.ipynb
```

### 4. Запуск API через Docker

```
docker-compose up --build
```

API будет доступно по адресу: http://localhost:8000/docs

## Данные

- `data/raw/` — исходные данные из CP1 (Kaggle: Global Central Bank Rates 1945-2026)
- `data/processed/` — подготовленные данные после очистки и feature engineering, используемые в CP2

## Результаты

| Модель | Macro F1 |
|--------|----------|
| Logistic Regression (baseline) | 0.138 |
| KNN | 0.377 |
| Random Forest | 0.999 |
| Gradient Boosting | 0.999 |
| CatBoost | 0.999 |
| Voting Classifier | 0.999 |

- Гиперпараметры подобраны для RandomForest и GradientBoosting (RandomizedSearchCV + TimeSeriesSplit)
- Выполнено уменьшение размерности с помощью PCA и визуализация
- Финальная модель: `models/final_model.pkl` (RandomForest, n_estimators=200, max_depth=6)

## API (FastAPI)

### Эндпоинты

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка работоспособности |
| GET | `/model-info` | Информация о модели и списке признаков |
| POST | `/predict` | Предсказание направления изменения ставки |

### Пример запроса

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "policy_rate_pct": 1.75,
    "year": 2024,
    "month": 5,
    "population": 67000000,
    "dataset_version": 1,
    "rate_change_bps": 25,
    "rate_jan2022_base": 0.5,
    "cumulative_change_since_2022_bps": 175,
    "rate_3m_avg": 1.5,
    "rate_12m_avg": 1.2,
    "rate_12m_max": 2.0,
    "rate_12m_min": 0.5,
    "is_all_time_high": 0,
    "is_all_time_low": 0,
    "gfc_period": 0,
    "covid_period": 0,
    "hike_cycle_2022_23": 1,
    "fed_rate": 5.25,
    "spread_vs_fed_bps": -350,
    "row_completeness_pct": 100,
    "policy_rate_pct_lag1": 1.5,
    "rate_3m_avg_lag1": 1.4,
    "rate_lag_1": 1.5,
    "rate_lag_3": 1.3,
    "rolling_mean_3": 1.45,
    "rolling_std_3": 0.1,
    "momentum_3": 0.25
  }'
```

### Пример ответа

```
{
  "predicted_class_index": 0,
  "predicted_class": "Hike",
  "probabilities": {
    "Hike": 0.92,
    "Hold": 0.07,
    "Cut": 0.01
  }
}
```

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)