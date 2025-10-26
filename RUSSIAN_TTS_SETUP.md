# GLaDOS Russian TTS Setup Guide

## Быстрая установка

### 1. Установка базовых зависимостей

```bash
# Установка с поддержкой CUDA
pip install -e ".[cuda,ru-full]"

# Или с CPU
pip install -e ".[cpu,ru-full]"
```

### 2. Проверка установки моделей

Убедитесь, что RVC модели находятся в правильной директории:

```bash
ls -la models/TTS/GLaDOS_ru/
```

Должны быть файлы:
- `ru_glados.pth` - RVC модель
- `added_IVF424_Flat_nprobe_1_ru_glados_v2.index` - индекс для FAISS

## Быстрый старт

### Вариант 1: Запуск полной системы GLaDOS на русском

```bash
# Запуск с русским конфигом
uv run glados start --config configs/glados_ru_config.yaml
```

Это запустит полную интерактивную систему:
- ✓ Распознавание речи (ASR)
- ✓ LLM для генерации ответов
- ✓ Русский голос GLaDOS (Silero TTS → RVC → Audio Processing)
- ✓ Полный аудио пайплайн с EQ/Compressor/Reverb

### Вариант 2: Только синтез речи (в коде)

```python
from glados.TTS import get_speech_synthesizer
import soundfile as sf

# Инициализация
tts = get_speech_synthesizer("glados_ru")

# Генерация речи
audio = tts.generate_speech_audio("Привет! Я GLaDOS.")

# Сохранение
sf.write("output.wav", audio, tts.sample_rate)
```

### Вариант 3: Запуск примеров

```bash
# Интерактивные примеры с разными вариантами использования
uv run python examples/glados_ru_example.py
```

## Компоненты системы

### 1. Silero V4 Russian TTS
- Модель: v4_ru
- Спикер: xenia
- Sample Rate: 48000 Hz
- Автоматическая расстановка ударений (put_accent=True)
- Конверсия е/ё (put_yo=True)

### 2. RVC Voice Conversion
- Модель: ru_glados.pth
- Индекс: added_IVF424_Flat_nprobe_1_ru_glados_v2.index
- Конверсия голоса в стиле GLaDOS

### 3. Audio Processing Pipeline
- **EQ**: 5-7 полосный параметрический эквалайзер
- **Compressor**: RMS компрессор с настраиваемыми attack/release
- **Reverb**: Алгоритмическая реверберация (Schroeder)

## Настройка параметров

### Через код

```python
from glados.TTS.tts_glados_ru import GLaDOSRuSynthesizer

glados = GLaDOSRuSynthesizer()

# Настройка EQ
glados.update_eq_band(2, gain_db=8.0, q=1.5)

# Настройка компрессора
glados.update_compressor(threshold_db=-15, ratio=6.0)

# Настройка ревербератора
glados.update_reverb(mix=0.5, decay_s=4.0)

# Сохранение настроек
glados.save_preset("my_settings")
```

### Через TUI

```python
from glados.audio_processing import run_audio_processor_tui

run_audio_processor_tui()
```

Клавиши управления:
- `q` - выход
- `s` - сохранить пресет
- `l` - загрузить пресет
- `r` - сброс к настройкам по умолчанию

## Пресеты

Доступны три встроенных пресета:

1. **glados_default** - стандартный пресет GLaDOS
2. **glados_subtle** - более естественный вариант
3. **glados_enhanced** - драматичный вариант с усиленной обработкой

### Создание пресетов

```python
from glados.audio_processing import PresetManager

pm = PresetManager()
pm.create_example_presets()

# Список доступных
presets = pm.list_presets()
print(presets)
```

## Отключение компонентов

```python
# Без RVC
glados = GLaDOSRuSynthesizer(enable_rvc=False)

# Без аудио-обработки
glados = GLaDOSRuSynthesizer(enable_audio_processing=False)

# Только Silero TTS
glados = GLaDOSRuSynthesizer(
    enable_rvc=False,
    enable_audio_processing=False
)
```

## Производительность

### GPU Acceleration

```python
# Использование CUDA
glados = GLaDOSRuSynthesizer(device="cuda")
```

### Типичные RTF (Real-Time Factor)

- **CPU (i7)**: RTF ≈ 0.3-0.5
- **GPU (RTX 3080)**: RTF ≈ 0.05-0.1

RTF < 1.0 означает генерацию быстрее реального времени.

## Структура файлов

```
src/glados/
├── audio_processing/
│   ├── __init__.py
│   ├── audio_processor.py      # EQ, Compressor, Reverb
│   ├── preset_manager.py       # Управление пресетами
│   ├── rvc_processor.py        # RVC voice conversion
│   └── audio_tui.py            # TUI для настройки
├── TTS/
│   ├── __init__.py
│   ├── tts_silero_ru.py        # Silero TTS
│   └── tts_glados_ru.py        # Полный пайплайн

examples/
└── glados_ru_example.py        # Примеры использования

docs/
└── GLADOS_RU_AUDIO_PROCESSING.md  # Полная документация
```

## Troubleshooting

### CUDA Out of Memory
```python
glados = GLaDOSRuSynthesizer(device="cpu")
```

### Модели не найдены
Убедитесь, что модели находятся в `models/TTS/GLaDOS_ru/`

### Медленная генерация
1. Используйте GPU: `device="cuda"`
2. Отключите RVC: `enable_rvc=False`
3. Упростите аудио-обработку

## Дополнительная документация

См. полную документацию в `docs/GLADOS_RU_AUDIO_PROCESSING.md`

## Примеры

Все примеры находятся в `examples/glados_ru_example.py`:

1. Базовое использование
2. Пользовательские пресеты
3. Настройка параметров в реальном времени
4. Пакетная обработка
5. Использование без RVC
6. Standalone аудио-обработка
7. Интерактивный TUI

## Зависимости

### Обязательные
- numpy
- scipy
- soundfile
- torch
- loguru
- textual
- pyyaml

### Опциональные
- faiss-cpu (для RVC индексации)
- librosa (для resampling)

## Лицензия

См. LICENSE в корне проекта.
