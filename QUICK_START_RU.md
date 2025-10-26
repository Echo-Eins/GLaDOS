# Быстрый запуск GLaDOS на русском языке

## 🚀 Автоматический запуск (РЕКОМЕНДУЕТСЯ)

### Windows (PowerShell)

```powershell
# Перейдите в папку проекта
cd D:\Coding\Python\GLaDOS

# Запустите скрипт
.\scripts\run_glados_ru.ps1
```

### Linux/Mac (Bash)

```bash
# Перейдите в папку проекта
cd ~/GLaDOS

# Сделайте скрипт исполняемым (один раз)
chmod +x scripts/run_glados_ru.sh

# Запустите скрипт
./scripts/run_glados_ru.sh
```

---

## 🔧 Ручной запуск (если нужен контроль)

### Windows (PowerShell)

```powershell
# 1. Перейдите в папку проекта
cd D:\Coding\Python\GLaDOS

# 2. Очистите Python cache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Recurse -File -Filter "*.pyo" | Remove-Item -Force

# 3. Очистите uv cache
uv cache clean

# 4. Очистите pip cache (опционально)
python -m pip cache purge

# 5. Синхронизируйте зависимости
uv sync --extra cuda --extra ru-full

# 6. Проверьте CUDA (опционально)
uv run python scripts/check_cuda.py

# 7. Запустите GLaDOS
uv run glados start --config configs/glados_ru_config.yaml
```

### Linux/Mac (Bash)

```bash
# 1. Перейдите в папку проекта
cd ~/GLaDOS

# 2. Очистите Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# 3. Очистите uv cache
uv cache clean

# 4. Очистите pip cache (опционально)
python -m pip cache purge

# 5. Синхронизируйте зависимости
uv sync --extra cuda --extra ru-full

# 6. Проверьте CUDA (опционально)
uv run python scripts/check_cuda.py

# 7. Запустите GLaDOS
uv run glados start --config configs/glados_ru_config.yaml
```

---

## 📝 Копируй-Вставляй команды (Windows)

**Вариант 1: Автоматический (один раз)**
```powershell
cd D:\Coding\Python\GLaDOS; .\scripts\run_glados_ru.ps1
```

**Вариант 2: Полная очистка + запуск**
```powershell
cd D:\Coding\Python\GLaDOS; Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force; Get-ChildItem -Recurse -File -Filter "*.pyc" | Remove-Item -Force; uv cache clean; uv sync --extra cuda --extra ru-full; uv run glados start --config configs/glados_ru_config.yaml
```

**Вариант 3: Быстрый запуск (без очистки)**
```powershell
cd D:\Coding\Python\GLaDOS; uv run glados start --config configs/glados_ru_config.yaml
```

---

## 📝 Копируй-Вставляй команды (Linux/Mac)

**Вариант 1: Автоматический (один раз)**
```bash
cd ~/GLaDOS && ./scripts/run_glados_ru.sh
```

**Вариант 2: Полная очистка + запуск**
```bash
cd ~/GLaDOS && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -type f -name "*.pyc" -delete 2>/dev/null; uv cache clean; uv sync --extra cuda --extra ru-full; uv run glados start --config configs/glados_ru_config.yaml
```

**Вариант 3: Быстрый запуск (без очистки)**
```bash
cd ~/GLaDOS && uv run glados start --config configs/glados_ru_config.yaml
```

---

## 🎯 Что делает скрипт?

1. **Очищает Python cache** - удаляет все `__pycache__`, `*.pyc`, `*.pyo`
2. **Очищает uv cache** - `uv cache clean`
3. **Очищает pip cache** - `pip cache purge`
4. **Синхронизирует зависимости** - `uv sync --extra cuda --extra ru-full`
5. **Проверяет CUDA** - показывает доступность GPU
6. **Запускает GLaDOS** - с русским конфигом

---

## 🔍 Проверка установки

```bash
# Проверка CUDA
uv run python scripts/check_cuda.py

# Проверка версии Python
python --version

# Проверка uv
uv --version

# Тест примеров
uv run python examples/glados_ru_example.py
```

---

## ❓ Troubleshooting

### Проблема: "NoneType has no attribute 'apply_tts'"

**Причина:** Старый кэш Python
**Решение:** Запустите полную очистку через скрипт

```powershell
# Windows
.\scripts\run_glados_ru.ps1

# Linux/Mac
./scripts/run_glados_ru.sh
```

### Проблема: Говорит по-английски

**Причина:** Не используется русский конфиг
**Решение:** Явно указывайте конфиг:

```bash
uv run glados start --config configs/glados_ru_config.yaml
```

### Проблема: CUDA not available

**Причина:** Нет GPU или не установлены драйверы
**Решение:**

1. Проверьте: `uv run python scripts/check_cuda.py`
2. Установите CUDA: https://developer.nvidia.com/cuda-toolkit
3. Или используйте CPU: `uv sync --extra cpu --extra ru-full`

### Проблема: Модуль не найден

**Причина:** Не синхронизированы зависимости
**Решение:**

```bash
uv sync --extra cuda --extra ru-full
```

---

## 📚 Дополнительная информация

- Полная документация: [RUSSIAN_TTS_SETUP.md](RUSSIAN_TTS_SETUP.md)
- Технические детали: [docs/GLADOS_RU_AUDIO_PROCESSING.md](docs/GLADOS_RU_AUDIO_PROCESSING.md)
- Примеры: [examples/glados_ru_example.py](examples/glados_ru_example.py)

---

## 🎮 Горячие клавиши

- `Ctrl+C` - остановить GLaDOS
- `q` - выход из TUI (если используется `glados tui`)

---

## 💡 Совет

**Создайте алиас для быстрого запуска:**

### Windows PowerShell (в профиле)

```powershell
# Откройте профиль PowerShell
notepad $PROFILE

# Добавьте алиас
function Start-GLaDOS-Ru {
    Set-Location "D:\Coding\Python\GLaDOS"
    .\scripts\run_glados_ru.ps1
}

# Теперь можно запускать из любой папки:
Start-GLaDOS-Ru
```

### Linux/Mac (.bashrc или .zshrc)

```bash
# Добавьте в ~/.bashrc или ~/.zshrc
alias glados-ru='cd ~/GLaDOS && ./scripts/run_glados_ru.sh'

# Теперь можно запускать из любой папки:
glados-ru
```
