# GLaDOS Russian Voice Synthesis with Audio Processing

Полный пайплайн для генерации русской речи GLaDOS с использованием Silero TTS, RVC конверсии голоса и продвинутой аудио-обработки.

## Архитектура

```
LLM Output (Text)
    ↓
Silero V4 Russian TTS (speaker: xenia, 48kHz)
    ↓
RVC Voice Conversion (GLaDOS voice model)
    ↓
Audio Processing Pipeline
    ├── Parametric EQ (5-7 bands)
    ├── RMS Compressor
    └── Schroeder Reverb
    ↓
Final Audio Output
```

## Установка зависимостей

```bash
# Основные зависимости
pip install torch torchaudio silero scipy soundfile librosa

# Опционально: для RVC с индексацией
pip install faiss-cpu  # или faiss-gpu для GPU

# Опционально: для YAML пресетов
pip install pyyaml
```

## Быстрый старт

### Базовое использование

```python
from glados.TTS.tts_glados_ru import GLaDOSRuSynthesizer

# Инициализация синтезатора
glados = GLaDOSRuSynthesizer()

# Генерация речи
text = "Привет! Я GLaDOS."
audio = glados.generate_speech_audio(text)

# Сохранение
import soundfile as sf
sf.write("output.wav", audio, glados.sample_rate)
```

### Использование через фабрику

```python
from glados.TTS import get_speech_synthesizer

# Получение синтезатора через фабрику
tts = get_speech_synthesizer("glados_ru")

audio = tts.generate_speech_audio("Добро пожаловать в лабораторию.")
```

## Модули

### 1. Audio Processing Pipeline

Модуль `audio_processor.py` содержит:

#### AudioEQProcessor
Параметрический эквалайзер на базе IIR biquad фильтров.

```python
from glados.audio_processing import AudioEQProcessor

eq = AudioEQProcessor(sample_rate=48000)

# Добавление полос
eq.add_band(freq=110, gain_db=0, q=0.7, filter_type='highpass')
eq.add_band(freq=3200, gain_db=5.0, q=1.2, filter_type='peak')
eq.add_band(freq=7000, gain_db=3.5, q=0.8, filter_type='highshelf')

# Применение
processed_audio = eq.apply(audio)
```

Типы фильтров:
- `lowpass` - низкочастотный фильтр
- `highpass` - высокочастотный фильтр
- `peak` - параметрический пик (boost/cut)
- `lowshelf` - низкочастотная полка
- `highshelf` - высокочастотная полка

#### Compressor
RMS-компрессор для контроля динамики.

```python
from glados.audio_processing import Compressor

comp = Compressor(
    sample_rate=48000,
    threshold_db=-20.0,
    ratio=4.0,
    attack_ms=10.0,
    release_ms=100.0,
    makeup_gain_db=3.0
)

processed_audio = comp.apply(audio)
```

#### Reverb
Алгоритмическая реверберация (Schroeder).

```python
from glados.audio_processing import Reverb

reverb = Reverb(
    sample_rate=48000,
    decay_s=2.0,
    pre_delay_ms=20.0,
    mix=0.3,
    damping=0.5,
    room_size=0.5
)

processed_audio = reverb.apply(audio)
```

#### AudioProcessingPipeline
Полный пайплайн обработки.

```python
from glados.audio_processing import AudioProcessingPipeline

pipeline = AudioProcessingPipeline(sample_rate=48000)

# Загрузка пресета GLaDOS
pipeline.load_glados_preset()

# Обработка
processed = pipeline.process(audio)

# Экспорт конфигурации
config = pipeline.to_dict()

# Импорт конфигурации
pipeline.from_dict(config)
```

### 2. Preset Manager

Управление пресетами в JSON/YAML.

```python
from glados.audio_processing import PresetManager

pm = PresetManager(presets_dir="./my_presets")

# Сохранение пресета
config = {
    "eq": [...],
    "compressor": {...},
    "reverb": {...}
}
pm.save_preset("my_preset", config, format="json")

# Загрузка пресета
config = pm.load_preset("my_preset")

# Список доступных пресетов
presets = pm.list_presets()

# Создание примеров пресетов
pm.create_example_presets()
```

### 3. RVC Processor

Voice conversion с использованием RVC моделей.

```python
from glados.audio_processing import create_rvc_processor

rvc = create_rvc_processor(
    model_path="models/TTS/GLaDOS_ru/ru_glados.pth",
    index_path="models/TTS/GLaDOS_ru/added_IVF424_Flat_nprobe_1_ru_glados_v2.index",
    simple_mode=True
)

converted_audio = rvc.process(audio, input_sample_rate=48000)
```

### 4. Silero TTS

Русский синтез речи с Silero V4.

```python
from glados.TTS.tts_silero_ru import SileroRuSynthesizer

tts = SileroRuSynthesizer(
    speaker="xenia",
    sample_rate=48000,
    put_accent=True,
    put_yo=True,
    device="cuda"  # или "cpu"
)

audio = tts.generate_speech_audio("Привет, мир!")
```

## Пресеты

### GLaDOS Default

Стандартный пресет для имитации голоса GLaDOS:

```json
{
  "eq": [
    {"type": "highpass", "frequency": 110, "gain_db": 0, "q_factor": 0.7},
    {"type": "peak", "frequency": 400, "gain_db": -2.0, "q_factor": 1.0},
    {"type": "peak", "frequency": 3200, "gain_db": 5.0, "q_factor": 1.2},
    {"type": "highshelf", "frequency": 7000, "gain_db": 3.5, "q_factor": 0.8}
  ],
  "compressor": {
    "threshold_db": -20,
    "ratio": 4.0,
    "attack_ms": 10,
    "release_ms": 100,
    "makeup_gain_db": 3
  },
  "reverb": {
    "decay_s": 3.0,
    "pre_delay_ms": 35,
    "mix": 0.35,
    "damping": 0.6,
    "room_size": 0.7
  }
}
```

### GLaDOS Subtle

Более естественный вариант с меньшей обработкой:

```json
{
  "eq": [
    {"type": "highpass", "frequency": 100, "gain_db": 0, "q_factor": 0.7},
    {"type": "peak", "frequency": 3000, "gain_db": 2.5, "q_factor": 1.0}
  ],
  "compressor": {
    "threshold_db": -25,
    "ratio": 2.5,
    "attack_ms": 15,
    "release_ms": 150,
    "makeup_gain_db": 2
  },
  "reverb": {
    "decay_s": 1.5,
    "pre_delay_ms": 20,
    "mix": 0.2,
    "damping": 0.5,
    "room_size": 0.5
  }
}
```

### GLaDOS Enhanced

Драматичный вариант с усиленной обработкой:

```json
{
  "eq": [
    {"type": "highpass", "frequency": 120, "gain_db": 0, "q_factor": 0.8},
    {"type": "peak", "frequency": 300, "gain_db": -3.0, "q_factor": 1.2},
    {"type": "peak", "frequency": 3500, "gain_db": 7.0, "q_factor": 1.5},
    {"type": "highshelf", "frequency": 8000, "gain_db": 5.0, "q_factor": 0.7}
  ],
  "compressor": {
    "threshold_db": -18,
    "ratio": 6.0,
    "attack_ms": 5,
    "release_ms": 80,
    "makeup_gain_db": 5
  },
  "reverb": {
    "decay_s": 4.5,
    "pre_delay_ms": 50,
    "mix": 0.45,
    "damping": 0.7,
    "room_size": 0.85
  }
}
```

## TUI (Text User Interface)

Интерактивный интерфейс для настройки параметров.

```python
from glados.audio_processing import run_audio_processor_tui, AudioProcessingPipeline

# Создание пайплайна
pipeline = AudioProcessingPipeline()
pipeline.load_glados_preset()

# Запуск TUI
run_audio_processor_tui(pipeline)
```

Клавиши управления:
- `q` - выход
- `s` - сохранить пресет
- `l` - загрузить пресет
- `r` - сбросить к настройкам по умолчанию

## Продвинутое использование

### Динамическая настройка параметров

```python
glados = GLaDOSRuSynthesizer()

# Настройка EQ
glados.update_eq_band(2, gain_db=8.0, q=1.5)

# Настройка компрессора
glados.update_compressor(
    threshold_db=-15,
    ratio=6.0,
    makeup_gain_db=4.0
)

# Настройка реверберации
glados.update_reverb(
    mix=0.5,
    decay_s=4.0,
    room_size=0.8
)

# Генерация с новыми настройками
audio = glados.generate_speech_audio("Текст с новыми настройками")

# Сохранение текущих настроек
glados.save_preset("my_custom_settings")
```

### Отключение компонентов

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

### Batch processing

```python
texts = [
    "Первая фраза",
    "Вторая фраза",
    "Третья фраза"
]

glados = GLaDOSRuSynthesizer()

for i, text in enumerate(texts):
    audio = glados.generate_speech_audio(text)
    sf.write(f"output_{i:03d}.wav", audio, glados.sample_rate)
```

## Оптимизация производительности

### GPU ускорение

```python
# Явное указание устройства
glados = GLaDOSRuSynthesizer(device="cuda")
```

### Кэширование моделей

Модели Silero кэшируются автоматически через torch.hub.

### RTF (Real-Time Factor)

Типичные значения RTF на различном оборудовании:

- CPU (Intel i7): RTF ≈ 0.3-0.5
- GPU (NVIDIA RTX 3080): RTF ≈ 0.05-0.1

RTF < 1.0 означает генерацию быстрее реального времени.

## Примеры

См. `examples/glados_ru_example.py` для полных примеров использования.

## Технические детали

### Пайплайн обработки

1. **Silero TTS**: Генерация базового аудио (48kHz, float32)
2. **RVC**: Конверсия голоса с использованием обученной модели
3. **EQ**: Коррекция частотной характеристики (5-7 полос)
4. **Compressor**: Контроль динамического диапазона
5. **Reverb**: Добавление пространства (алгоритм Schroeder)

### Формат аудио

- Sample Rate: 48000 Hz
- Bit Depth: float32
- Channels: Mono

### Алгоритмы

- **EQ**: IIR biquad фильтры (2-го порядка)
- **Compressor**: RMS envelope follower с переменными attack/release
- **Reverb**: Параллельные comb + последовательные allpass фильтры

## Troubleshooting

### CUDA Out of Memory

Используйте CPU:
```python
glados = GLaDOSRuSynthesizer(device="cpu")
```

### Медленная генерация

1. Используйте GPU
2. Отключите RVC: `enable_rvc=False`
3. Упростите аудио-обработку

### Качество звука

Для лучшего качества:
1. Используйте полный пайплайн (Silero + RVC + Audio Processing)
2. Экспериментируйте с пресетами
3. Настройте параметры через TUI

## API Reference

См. docstrings в исходном коде для полного API reference.

## Лицензия

См. LICENSE в корне проекта.
