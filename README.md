# concepts

Исследование представления понятия в трансформере: программа, код проверки и итоги.

- `concepts.tex` / `concepts.pdf` — программа исследования и предварительные результаты (LaTeX, рус.).
- `SESSION_SUMMARY.md` — итоги сессии: что сделано, числа, выводы, как воспроизвести.
- `analysis/` — код (`uv` + `torch[cpu]` + `transformers`): инварианты, метрики, пробники и эксперименты.

## Сборка документа

```
latexmk -pdf concepts.tex
```

## Анализ

```
cd analysis
uv sync
uv run pytest -q

# примеры прогонов
uv run python scripts/run_h1.py
uv run python scripts/run_related.py
uv run python scripts/run_trajectory.py
```

Результаты сохраняются в `analysis/results/*.json`.
