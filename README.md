# MultiverseTester

Симуляция пригодности вселенных для жизни при различных значениях фундаментальных физических констант. Исследует пространство параметров мультивселенной и вычисляет индекс пригодности для жизни.

**Автор:** Timur Isanov  
**Email:** tisanov@yahoo.com

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
```

### Запуск скриптов напрямую

```bash
python main.py           # Основной анализ
python 2Doptimizator.py  # 2D оптимизация
python 3doptimizator.py  # 3D ландшафт
python 4Doptimizator.py  # 4D гиперобъём
python 5Doptimizator.py  # 5D
python 6D_optimizator.py # 6D
```

## Модель

- **Атомная физика:** радиус Бора, энергия Ридберга, комптоновская длина волны
- **Ядерная физика:** энергия связи (формула Вайцзеккера), кулоновский барьер
- **Звёздный нуклеосинтез:** pp-цепочка, CNO-цикл, тройная альфа, s/r-процессы, сверхновые
- **Индекс пригодности:** DEAD → HOSTILE → MARGINAL → HABITABLE → OPTIMAL

## Зависимости

- Python >= 3.8
- numpy
- matplotlib
- scipy
- scikit-image (опционально, для изоповерхностей в 3D)

## Лицензия

MIT
