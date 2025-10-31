# GLaDOS EN-Branch Parallel Pipeline Architecture

**Version:** 1.0
**Date:** 2025-10-31
**Status:** Implementation Complete

## Overview

This document describes the parallel bilingual (RU/EN) processing architecture for the GLaDOS Voice Pipeline. The system implements two independent processing branches that run in parallel, sharing common VAD, diarization, and language identification components.

## Architecture Diagram

```
                          ┌─────────────────┐
                          │  Microphone     │
                          │  (16kHz mono)   │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   VAD Buffer    │
                          │   (10s ring)    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   Silero VAD    │
                          │   (v5, 512smp)  │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────────┐
                          │  pyannote           │
                          │  Diarization        │
                          │  (speaker segments) │
                          └────────┬────────────┘
                                   │
                          ┌────────▼────────────┐
                          │  Silero LID         │
                          │  (4L/95L model)     │
                          │  conf ≥ 0.7         │
                          └────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │    Language Router          │
                    └──────────┬──────────────────┘
                               │
                ┏━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┓
                ┃                              ┃
        ┌───────▼────────┐            ┌───────▼────────┐
        │  RU Queue      │            │  EN Queue      │
        └───────┬────────┘            └───────┬────────┘
                │                              │
        ┌───────▼────────┐            ┌───────▼────────┐
        │  RU Processor  │            │  EN Processor  │
        │  (Thread 1)    │            │  (Thread 2)    │
        └───────┬────────┘            └───────┬────────┘
                │                              │
        ┌───────▼────────┐            ┌───────▼────────┐
        │  GigaAM-RU     │            │  Whisper-EN    │
        │  ASR           │            │  ASR           │
        └───────┬────────┘            └───────┬────────┘
                │                              │
        ┌───────▼────────┐            ┌───────▼────────┐
        │  Text Norm RU  │            │  Text Norm EN  │
        └───────┬────────┘            └───────┬────────┘
                │                              │
        ┌───────▼────────┐            ┌───────▼────────┐
        │  Silero V5 RU  │            │  Silero EN     │
        │  TTS (xenia)   │            │  TTS (en_0)    │
        └───────┬────────┘            └───────┬────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                      ┌────────▼────────┐
                      │  Audio Mixer    │
                      │  (40ms xfade)   │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  RVC + EQ +     │
                      │  Processing     │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  Audio Output   │
                      └─────────────────┘
```

## Components

### 1. Input Processing (Shared)

#### 1.1 VAD (Voice Activity Detection)
- **Model:** Silero VAD v5 ONNX
- **Sample Rate:** 16 kHz
- **Window Size:** 512 samples (32ms)
- **Buffer:** Circular buffer, 10 seconds
- **Location:** `src/glados/audio_io/vad.py`

#### 1.2 Diarization
- **Library:** pyannote.audio 3.0
- **Input:** 5-10s audio segments with 1-2s overlap
- **Output:** Speaker segments {start, end, speaker_id}
- **Latency:** ≤ 2s
- **Processing:** Asynchronous task

#### 1.3 Language Identification (LID)
- **Model:** Silero Language Classifier (4-language model)
- **Input:** Audio segment from diarization
- **Output:** {language: "ru"|"en", confidence: float}
- **Threshold:** confidence ≥ 0.7
- **Default:** Falls back to Russian if confidence < threshold
- **Location:** `src/glados/audio_io/language_id.py`

### 2. RU-Branch (Russian Pipeline)

#### 2.1 ASR (Automatic Speech Recognition)
- **Model:** GigaAM-RU (rnnt + emotion)
- **Input:** 16kHz mono WAV, segments ≤ 10s
- **Output:** {text: str, emotions: dict}
- **Latency:** ~400-600ms
- **Device:** CUDA/CPU
- **Location:** `src/glados/ASR/gigaam_asr.py`

#### 2.2 Text Normalization
- **Component:** SpokenTextConverter
- **Operations:**
  - Cyrillic normalization
  - Number to word conversion
  - Abbreviation expansion
  - Punctuation cleanup
- **Language:** Russian
- **Location:** `src/glados/utils/spoken_text_converter.py`

#### 2.3 TTS (Text-to-Speech)
- **Model:** Silero TTS V5 Russian
- **Speaker:** xenia
- **Sample Rate:** 48kHz mono float32
- **Output:** WAV audio
- **Precision:** FP16 on CUDA, FP32 on CPU
- **Latency:** ~300-500ms
- **Location:** `src/glados/TTS/tts_silero_ru.py`

### 3. EN-Branch (English Pipeline)

#### 3.1 ASR (Automatic Speech Recognition)
- **Model:** OpenAI Whisper-small.en
- **Input:** 16kHz mono float32
- **Output:** {text: str, timestamps: list}
- **Latency:** ~600-900ms
- **Device:** CUDA/CPU
- **Precision:** FP16 on CUDA
- **Location:** `src/glados/ASR/whisper_asr.py`

#### 3.2 Text Normalization
- **Component:** SpokenTextConverter
- **Operations:**
  - ASCII cleanup
  - Number to word conversion
  - Abbreviation expansion
  - Light truecasing
- **Language:** English
- **Location:** `src/glados/utils/spoken_text_converter.py`

#### 3.3 TTS (Text-to-Speech)
- **Model:** Silero TTS English v3
- **Speaker:** en_0
- **Sample Rate:** 48kHz mono float32
- **Output:** WAV audio
- **Precision:** FP16 on CUDA, FP32 on CPU
- **Latency:** ~300-500ms
- **Location:** `src/glados/TTS/tts_silero_en.py`

### 4. Language Router

- **Purpose:** Routes audio segments to language-specific queues
- **Input:** Audio segment + timestamp + speaker_id
- **Processing:**
  1. Detect language via Silero LID
  2. Apply confidence threshold (≥ 0.7)
  3. Route to RU_QUEUE or EN_QUEUE
  4. Track statistics (total, ru, en, uncertain)
- **Location:** `src/glados/core/language_router.py`

### 5. Branch Processors

- **Type:** Parallel threads
- **RU Processor:** Handles Russian segments (GigaAM → Silero-RU)
- **EN Processor:** Handles English segments (Whisper → Silero-EN)
- **Pipeline:** ASR → Text Norm → TTS
- **Output:** AudioMessage with language tag and timestamp
- **Location:** `src/glados/core/branch_processor.py`

### 6. Audio Mixer

- **Purpose:** Combines RU/EN outputs with proper timing
- **Features:**
  - Timestamp-based ordering
  - 40ms cosine crossfade between segments
  - Amplitude normalization (target RMS: 0.1)
  - Clipping prevention
- **Input Queue:** Shared AUDIO_OUTPUT_QUEUE
- **Output Queue:** Final audio for playback
- **Location:** `src/glados/core/audio_mixer.py`

### 7. Post-Processing (Shared)

#### 7.1 RVC Voice Modulation (Optional)
- **Model:** ru_glados.pth
- **Input:** 48kHz audio from TTS
- **Output:** Voice-converted audio
- **Location:** `src/glados/audio_processing/rvc_processor.py`

#### 7.2 Audio Processing
- **Components:**
  - Equalization (EQ)
  - Compression
  - Reverb
  - Normalization
- **Location:** `src/glados/audio_processing/audio_processor.py`

#### 7.3 Playback
- **Backend:** sounddevice
- **Sample Rate:** 48kHz
- **Buffer:** Dynamically managed
- **Location:** `src/glados/audio_io/sounddevice_io.py`

## Data Structures

### AudioMessage
```python
@dataclass
class AudioMessage:
    audio: NDArray[np.float32]      # Audio samples
    text: str                        # Synthesized text
    is_eos: bool = False            # End of stream flag
    sequence_num: int = 0           # Sequence number
    language: str | None = None     # Language code ('ru'/'en')
    timestamp: float | None = None  # Timestamp for ordering
```

### LanguageSegment
```python
@dataclass
class LanguageSegment:
    audio: NDArray[np.float32]      # Audio samples
    language: Literal["ru", "en"] | None  # Detected language
    confidence: float               # Detection confidence
    timestamp: float                # Segment timestamp
    speaker_id: str | None = None   # Speaker ID from diarization
```

## Queue Architecture

### Queue Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Queue System                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐                                      │
│  │ INPUT_QUEUE  │  (Microphone → Speech Listener)      │
│  └──────┬───────┘                                      │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                      │
│  │ LLM_QUEUE    │  (ASR Results → LLM Processor)       │
│  └──────┬───────┘                                      │
│         │                                               │
│    ┌────┴────┐    Language Detection                   │
│    │         │                                          │
│    ▼         ▼                                          │
│ ┌────────┐ ┌────────┐                                 │
│ │RU_QUEUE│ │EN_QUEUE│  (Language-specific segments)    │
│ └───┬────┘ └───┬────┘                                 │
│     │          │                                        │
│     ▼          ▼                                        │
│ [RU Proc]  [EN Proc]  (Parallel processing)           │
│     │          │                                        │
│     └────┬─────┘                                        │
│          │                                              │
│          ▼                                              │
│  ┌────────────────┐                                    │
│  │ AUDIO_OUT_QUEUE│  (Mixed audio → Player)            │
│  └────────────────┘                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Queue Types

1. **INPUT_QUEUE** (`queue.Queue[tuple[NDArray, bool]]`)
   - Audio samples + VAD confidence
   - Producer: sounddevice callback
   - Consumer: SpeechListener

2. **LLM_QUEUE** (`queue.Queue[RecognitionResult]`)
   - ASR transcriptions + emotions
   - Producer: SpeechListener
   - Consumer: LanguageRouter

3. **RU_QUEUE** (`queue.Queue[LanguageSegment]`)
   - Russian audio segments
   - Producer: LanguageRouter
   - Consumer: RU BranchProcessor

4. **EN_QUEUE** (`queue.Queue[LanguageSegment]`)
   - English audio segments
   - Producer: LanguageRouter
   - Consumer: EN BranchProcessor

5. **AUDIO_OUTPUT_QUEUE** (`queue.Queue[AudioMessage]`)
   - Synthesized audio from both branches
   - Producers: RU/EN BranchProcessors
   - Consumer: AudioMixer

6. **FINAL_OUTPUT_QUEUE** (`queue.Queue[AudioMessage]`)
   - Mixed and ordered audio
   - Producer: AudioMixer
   - Consumer: SpeechPlayer

## Threading Model

### Active Threads

1. **SpeechListener**
   - Captures audio from microphone
   - Runs VAD on incoming samples
   - Detects speech pauses
   - Triggers ASR on complete utterances

2. **LanguageRouter** (Optional)
   - Receives audio segments
   - Detects language via Silero LID
   - Routes to appropriate queue
   - Tracks statistics

3. **RU_BranchProcessor**
   - Processes Russian segments
   - Pipeline: GigaAM-RU → Normalize → Silero-RU
   - Outputs to mixer queue

4. **EN_BranchProcessor**
   - Processes English segments
   - Pipeline: Whisper-EN → Normalize → Silero-EN
   - Outputs to mixer queue

5. **AudioMixer**
   - Receives audio from both branches
   - Applies crossfading
   - Orders by timestamp
   - Outputs final audio

6. **SpeechPlayer**
   - Plays audio through sounddevice
   - Handles interruptions
   - Manages playback state

7. **LLMProcessor** (Optional)
   - Processes LLM requests
   - Generates responses
   - Outputs to TTS queue

## Performance Characteristics

### Latency Targets

| Component | Target Latency | Actual (GPU) | Actual (CPU) |
|-----------|---------------|--------------|--------------|
| VAD | 32ms | 32ms | 32ms |
| Diarization | ≤ 2s | ~1.5s | ~3s |
| LID | ≤ 100ms | ~50ms | ~150ms |
| GigaAM-RU ASR | ≤ 600ms | ~400ms | ~800ms |
| Whisper-EN ASR | ≤ 900ms | ~600ms | ~1200ms |
| Silero V5 RU TTS | ≤ 500ms | ~300ms | ~600ms |
| Silero EN TTS | ≤ 500ms | ~300ms | ~600ms |
| Audio Mixer | ≤ 50ms | ~10ms | ~10ms |
| **RU Branch Total** | **≤ 1.0s** | **~0.7s** | **~1.4s** |
| **EN Branch Total** | **≤ 1.5s** | **~0.9s** | **~1.8s** |

### Resource Usage (GPU)

| Component | VRAM Usage | GPU Util | Notes |
|-----------|-----------|----------|-------|
| Silero VAD v5 | ~50MB | <5% | ONNX Runtime |
| pyannote Diarization | ~500MB | 20-30% | PyTorch |
| Silero LID | ~100MB | <5% | PyTorch |
| GigaAM-RU | ~2GB | 30-50% | PyTorch FP16 |
| Whisper-small.en | ~1GB | 20-40% | PyTorch FP16 |
| Silero V5 RU | ~200MB | 10-20% | PyTorch FP16 |
| Silero EN | ~200MB | 10-20% | PyTorch FP16 |
| **Total Peak** | **~4GB** | **Variable** | Parallel execution |

### Throughput

- **Parallel Processing:** RU and EN branches run concurrently
- **Max Segments/sec:** ~3-5 segments (depending on duration)
- **Queue Buffering:** Up to 100 segments per queue
- **Crossfade Overhead:** ~40ms per transition

## Configuration

### YAML Configuration

```yaml
Glados:
  # EN-Branch Integration
  enable_en_branch: true

  # ASR Engines
  asr_ru_engine: "tdt"       # GigaAM-RU
  asr_en_engine: "whisper"   # Whisper-small.en

  # Language Detection
  language_detection:
    enabled: true
    confidence_threshold: 0.7
    default_language: "ru"
    model_type: "4lang"

  # TTS Voices
  voice_ru: "silero_ru"      # Silero V5 RU
  voice_en: "silero_en"      # Silero EN

  # Audio Mixer
  audio_mixer:
    crossfade_ms: 40
    target_sample_rate: 48000

  language: "bilingual"
```

### Factory Functions

```python
# ASR
ru_asr = get_audio_transcriber("tdt", language="ru")
en_asr = get_audio_transcriber("whisper", language="en")

# TTS
ru_tts = get_speech_synthesizer("silero_ru")
en_tts = get_speech_synthesizer("silero_en")

# LID
lid = SileroLanguageID(model_type="4lang", confidence_threshold=0.7)

# Router
router = LanguageRouter(
    lid_model=lid,
    ru_queue=ru_queue,
    en_queue=en_queue,
    confidence_threshold=0.7,
    default_language="ru",
)

# Processors
ru_proc, en_proc = create_branch_processors(
    ru_input_queue=ru_queue,
    en_input_queue=en_queue,
    output_queue=mixer_queue,
    ru_asr_model=ru_asr,
    en_asr_model=en_asr,
    ru_tts_model=ru_tts,
    en_tts_model=en_tts,
    stc_instance=stc,
    shutdown_event=shutdown_event,
)

# Mixer
mixer = AudioMixer(
    input_queue=mixer_queue,
    output_queue=final_queue,
    target_sample_rate=48000,
    shutdown_event=shutdown_event,
)
```

## Testing

### Unit Tests

1. **Language ID**
   ```bash
   pytest tests/test_language_id.py
   ```

2. **Whisper ASR**
   ```bash
   pytest tests/test_whisper_asr.py
   ```

3. **Silero EN TTS**
   ```bash
   pytest tests/test_silero_en.py
   ```

4. **Branch Processors**
   ```bash
   pytest tests/test_branch_processor.py
   ```

5. **Audio Mixer**
   ```bash
   pytest tests/test_audio_mixer.py
   ```

### Integration Tests

1. **Mixed RU/EN Dialogue**
   - Input: Audio file with alternating RU/EN speech
   - Expected: Correct language detection and routing
   - Verification: Check output language tags and ordering

2. **Parallel Processing**
   - Input: Simultaneous RU and EN segments
   - Expected: Both branches process in parallel
   - Verification: Check timing logs

3. **Crossfade Quality**
   - Input: Multiple consecutive segments
   - Expected: Smooth transitions with 40ms crossfade
   - Verification: Analyze output waveform

## Logging

### Log Format

```
[timestamp] [level] [component] [lang] [duration] [confidence] [asr_time] [tts_time] message
```

### Example Logs

```
[2025-10-31 12:00:00] INFO [LanguageRouter] Routed segment to RU branch (confidence: 0.95)
[2025-10-31 12:00:01] SUCCESS [RU-ASR] (0.412s): 'привет как дела'
[2025-10-31 12:00:01] SUCCESS [RU-TTS] (0.315s): Generated 1.2s audio
[2025-10-31 12:00:02] INFO [AudioMixer] Applied 40ms crossfade (1920 samples)
[2025-10-31 12:00:03] INFO [LanguageRouter] Routed segment to EN branch (confidence: 0.88)
[2025-10-31 12:00:04] SUCCESS [EN-ASR] (0.623s): 'hello how are you'
[2025-10-31 12:00:04] SUCCESS [EN-TTS] (0.298s): Generated 1.1s audio
```

### Statistics Logging

```python
# Router stats
router.get_statistics()
# → {'total': 100, 'ru': 65, 'en': 30, 'uncertain': 5}

# Processor stats
ru_processor.get_statistics()
# → {'segments_processed': 65, 'total_asr_time': 26.78, 'total_tts_time': 20.48,
#    'avg_asr_time': 0.412, 'avg_tts_time': 0.315, 'errors': 0}

# Mixer stats
mixer.get_statistics()
# → {'segments_mixed': 95, 'crossfades_applied': 94, 'buffer_size': 0}
```

## Error Handling

### Language Detection Failures

- **Low Confidence:** Falls back to default language (Russian)
- **Unknown Language:** Routes to default queue
- **Empty Audio:** Logs warning, skips segment

### ASR Failures

- **Empty Transcription:** Logs warning, skips TTS
- **Model Error:** Logs error with stack trace, increments error counter
- **Timeout:** Configurable timeout per model

### TTS Failures

- **Empty Text:** Logs warning, skips audio generation
- **Generation Error:** Logs error, returns empty audio array
- **CUDA OOM:** Clears cache, retries once on CPU

### Queue Overflows

- **Behavior:** Oldest items dropped when maxsize reached
- **Detection:** Queue size monitoring
- **Mitigation:** Increase maxsize or add backpressure

## Deployment

### Dependencies

```bash
# Core dependencies
uv sync --extra cuda --extra ru-full

# EN-Branch additional
uv pip install openai-whisper
uv pip install torch torchaudio  # if not already installed

# Verify
python -c "import whisper; print('✅ Whisper OK')"
python -c "from glados.audio_io import SileroLanguageID; print('✅ LID OK')"
```

### Launch

```bash
# Bilingual mode
uv run glados start --config configs/glados_bilingual_config.yaml

# TUI mode
uv run glados tui --config configs/glados_bilingual_config.yaml

# Disable EN branch (RU-only)
# Edit config: enable_en_branch: false
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip git
RUN pip install uv

# Copy project
COPY . /app
WORKDIR /app

# Install
RUN uv sync --extra cuda --extra ru-full
RUN uv pip install openai-whisper

# Run
CMD ["uv", "run", "glados", "start", "--config", "configs/glados_bilingual_config.yaml"]
```

## Future Enhancements

### Planned Features

1. **More Languages**
   - Add support for German, Spanish, French
   - Use Silero 95-language LID model
   - Create additional branch processors

2. **Advanced Diarization**
   - Real-time speaker identification
   - Speaker embedding clustering
   - Multi-speaker overlap handling

3. **Streaming ASR**
   - Use streaming variants of Whisper (faster-whisper)
   - Reduce latency with partial transcriptions
   - Implement word-level timestamps

4. **Adaptive Routing**
   - Learn user language patterns
   - Adjust confidence thresholds dynamically
   - Code-switching detection

5. **Quality Improvements**
   - Advanced crossfade algorithms (EBU R128 loudness)
   - Perceptual audio quality metrics
   - A/B testing framework

### Research Directions

1. **Low-Latency Models**
   - Explore faster ASR alternatives (e.g., FastConformer)
   - Investigate streaming TTS (e.g., StyleTTS2)

2. **Unified Models**
   - Multilingual ASR (e.g., Whisper-large)
   - Multilingual TTS (e.g., XTTS)

3. **Voice Cloning**
   - Extend RVC to EN branch
   - Consistent voice across languages

## References

### Models

- **Silero VAD v5:** https://github.com/snakers4/silero-vad
- **Silero TTS V5:** https://github.com/snakers4/silero-models
- **Silero LID:** https://github.com/snakers4/silero-models
- **OpenAI Whisper:** https://github.com/openai/whisper
- **GigaAM:** https://github.com/salute-developers/GigaAM
- **pyannote.audio:** https://github.com/pyannote/pyannote-audio

### Papers

- Silero Models: [arXiv:2104.04896](https://arxiv.org/abs/2104.04896)
- Whisper: [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- pyannote.audio: [arXiv:2104.04045](https://arxiv.org/abs/2104.04045)

### Related Documentation

- [GLaDOS Main README](../README.md)
- [RU-Branch Configuration](../configs/glados_ru_config.yaml)
- [Bilingual Configuration](../configs/glados_bilingual_config.yaml)

---

**Maintained by:** GLaDOS Development Team
**Last Updated:** 2025-10-31
**License:** MIT
