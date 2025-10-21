import torch
import gigaam

# === Настройки ===
AUDIO_PATH = r"C:\Users\Lenovo\Downloads\ex4.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Используется устройство: {DEVICE.upper()}")

# === Загрузка моделей ===
print("Загружаем модели GigaAM...")
asr_model = gigaam.load_model("rnnt")
emo_model = gigaam.load_model("emo")

# Принудительно задаём устройство
asr_model.device = DEVICE
emo_model.device = DEVICE

# === Расшифровка речи ===
print("→ Распознаем речь...")
text = asr_model.transcribe(AUDIO_PATH)
print(f"🗣️ Распознанный текст: {text}")

# === Анализ эмоций ===
print("→ Анализируем эмоцию...")
emotions = {k: round(float(v), 4) for k, v in emo_model.get_probs(AUDIO_PATH).items()}
dominant = max(emotions, key=emotions.get)

print("🎭 Эмоции:")
for emo, val in emotions.items():
    print(f"  {emo:<8}: {val:.4f}")
print(f"➡️ Основная эмоция: {dominant}")

result = {"text": text, "emotion": dominant, "emotions": emotions}

print("\n📦 Итоговый результат:")
print(result)
