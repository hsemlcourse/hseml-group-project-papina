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
├── presentation
├── report
│   └── report.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .flake8
└── README.md
```

## Быстрый старт

```bash
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Запуск ноутбука

```bash
jupyter notebook notebooks/cp2.ipynb
```

### Запуск через Docker

```bash
docker compose up --build
```

После запуска откройте `http://localhost:8888`.

## Данные

- `data/raw/` — исходные данные из CP1.
- `data/processed/` — подготовленные данные, используемые в CP2.

## Результаты

- Модели: LogisticRegression, RandomForest, KNN, CatBoost, GradientBoosting, VotingClassifier.
- Гиперпараметры подобраны минимум для двух моделей (`RandomForest`, `GradientBoosting`).
- Выполнено уменьшение размерности с помощью PCA и визуализация.
- Финальная модель: `models/final_model.pkl`.

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
