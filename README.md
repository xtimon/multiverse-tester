# MultiverseTester

Симуляция пригодности вселенных для жизни при различных значениях фундаментальных физических констант. Исследует пространство параметров мультивселенной (от 2D до 8D) и вычисляет индекс пригодности.

**Закон 8D:** допустимый диапазон изменения константы обратно пропорционален силе соответствующего взаимодействия — ΔP ∝ 1/F.

## Установка

```bash
pip install .
```

С опциональной поддержкой изоповерхностей (scikit-image):

```bash
pip install ".[full]"
```

### Сборка для публикации на PyPI

```bash
pip install build
python -m build
# Файлы будут в dist/
pip install twine
twine upload dist/*
```

**GitHub Actions:** CI запускается на push/PR. Публикация на PyPI — при создании release или push тега `v*`. Добавьте `PYPI_API_TOKEN` в Secrets репозитория.

## Использование

### Программный интерфейс

```python
from multiverse_tester import (
    UniverseParameters,
    UniverseAnalyzer,
    UniversalConstants,
    HabitabilityIndex,
)

# Создание вселенной с заданными параметрами
u = UniverseParameters(
    name="Тестовая вселенная",
    alpha=1/137.036,  # постоянная тонкой структуры
    m_p=1.6726219e-27,  # масса протона (кг)
)

# Анализ пригодности для жизни
analyzer = UniverseAnalyzer(u)
index, score, metrics = analyzer.calculate_habitability_index()

print(f"Индекс пригодности: {score:.3f}")
print(f"Категория: {index.name}")
```

### CLI

```bash
# Основной анализ
multiverse-analyze

# 2D оптимизация (α, m_p)
multiverse-optimize-2d

# 3D ландшафт (α, m_p, m_e)
multiverse-optimize-3d

# 4D гиперобъём (α, m_p, m_e, G)
multiverse-optimize-4d

# 5D (α, m_p, m_e, G, c)
multiverse-optimize-5d

# 6D (α, m_p, m_e, G, c, ħ)
multiverse-optimize-6d

# 7D (α, m_p, m_e, G, c, ħ, ε₀)
multiverse-optimize-7d

# 8D (α, m_p, m_e, G, c, ħ, ε₀, k_B)
multiverse-optimize-8d

# 9D (α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀)
multiverse-optimize-9d

# 10D (+ Λ)
multiverse-optimize-10d
```

### Интерактивное веб-демо (Streamlit)

```bash
pip install streamlit   # или: pip install ".[demo]"
streamlit run streamlit_demo.py
```

Откройте браузер и исследуйте «пузырь жизни» — меняйте ползунки (α, m_p, m_e, G, c, ħ, ε₀) и наблюдайте, как меняется пригодность вселенной. Ландшафт показывает область пригодности в плоскости (α, m_p).

### Пакетный запуск всех оптимизаторов

```bash
python run_all_optimizers.py   # 2D→10D, отчёт в reports/
```

### Запуск скриптов напрямую

```bash
python main.py           # Основной анализ
python 2Doptimizator.py  # 2D оптимизация
python 3Doptimizator.py  # 3D ландшафт
python 4Doptimizator.py  # 4D гиперобъём
python 5Doptimizator.py  # 5D
python 6D_optimizator.py # 6D
python 7D_optimizator.py # 7D (α, m_p, m_e, G, c, ħ, ε₀)
python 8D_optimizator.py # 8D (α, m_p, m_e, G, c, ħ, ε₀, k_B)
python 9D_optimizator.py  # 9D (+ H₀)
python 10D_optimizator.py # 10D (+ Λ)
```

## Отчёты

| Файл | Описание |
|------|----------|
| [reports/OPTIMIZATION_REPORT.md](reports/OPTIMIZATION_REPORT.md) | Сводная таблица 2D–8D |
| [reports/FULL_ANALYSIS_2D_TO_8D.md](reports/FULL_ANALYSIS_2D_TO_8D.md) | Полный анализ, Закон 8D, иерархия констант |
| [reports/MATHEMATICAL_FORMALIZATION.md](reports/MATHEMATICAL_FORMALIZATION.md) | Математическая формализация: ΔP(F) = min(20, 2.0·F^(-0.15)) |

## Модель

- **Атомная физика:** радиус Бора, энергия Ридберга, комптоновская длина волны
- **Ядерная физика:** энергия связи (формула Вайцзеккера), кулоновский барьер
- **Звёздный нуклеосинтез:** pp-цепочка, CNO-цикл, тройная альфа, s/r-процессы, сверхновые
- **Индекс пригодности:** DEAD → HOSTILE → MARGINAL → HABITABLE → OPTIMAL

## Результаты (2D→9D)

- **α** — единственный параметр с чётким оптимумом (~0.007–0.011)
- **m_p, m_e, ε₀, k_B** — стабилизируются на ~0.1× (могут быть в 10 раз меньше)
- **c, ħ, H₀** — стабилизируются на ~0.2× (могут быть в 5 раз меньше)
- **G** — стабилизируется на ~0.05× (может быть в 20 раз слабее)
- **Универсальная формула:** ΔP(F) = min(20, 2.0·F^(-0.15))

## Зависимости

- Python >= 3.8
- numpy
- matplotlib
- scipy
- scikit-image (опционально, для изоповерхностей в 3D)

## Лицензия

MIT

**Автор:** Timur Isanov
**Email:** tisanov@yahoo.com
