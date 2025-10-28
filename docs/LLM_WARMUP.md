# LLM Warmup and Keep-Alive Mechanism

## Overview

The GLaDOS system now includes an intelligent LLM warmup mechanism that pre-loads the language model into Ollama memory during system initialization. This eliminates the 1-2 minute delay that typically occurs on the first user request.

## Key Features

### 1. Automatic Model Pre-loading
- **When**: During GLaDOS initialization (after ASR warmup)
- **How**: Sends minimal request to `/api/generate` endpoint
- **Result**: Model is loaded and ready before first user interaction

### 2. Keep-Alive Management
- **Parameter**: `keep_alive_timeout` (default: "30m")
- **Purpose**: Controls how long Ollama keeps the model in memory
- **Behavior**: Automatically extends on each request

### 3. Graceful Error Handling
- Non-blocking: System continues even if warmup fails
- Detailed logging: Clear messages about warmup status
- Timeout protection: Won't hang system startup

## Configuration

### YAML Configuration

Add to your `glados_config.yaml`:

```yaml
Glados:
  llm_model: "huihui_ai/qwen2.5-abliterate:14b"
  completion_url: "http://localhost:11434/api/chat"
  keep_alive_timeout: "30m"  # How long to keep LLM loaded
  # ... other settings
```

### Supported Time Formats

- `"5m"` - 5 minutes
- `"30m"` - 30 minutes (recommended)
- `"1h"` - 1 hour
- `"2h"` - 2 hours
- `0` - Unload immediately after request

### Choosing the Right Timeout

**Short timeout (5m):**
- ✅ Lower memory usage
- ❌ Model reloads frequently
- **Use case**: Limited VRAM, infrequent usage

**Medium timeout (30m - default):**
- ✅ Good balance
- ✅ Handles typical conversation gaps
- **Use case**: Most users

**Long timeout (1-2h):**
- ✅ Model always ready
- ❌ Higher memory usage
- **Use case**: Heavy usage, large VRAM

## How It Works

### Initialization Sequence

```
1. System starts
2. ASR model loads + warmup (0.wav file)
3. LLM warmup triggered:
   - POST to /api/generate with empty prompt
   - Ollama loads model into VRAM
   - keep_alive timer starts
4. TTS/RVC models load
5. System ready

Total time: ASR warmup (1s) + LLM load (30-120s) + TTS load (2s)
```

### Request Lifecycle

```
User speaks → STT transcribes → LLM processes → TTS synthesizes → Audio plays
              (instant)          (instant!)      (2-3s)           (streaming)
```

**Without warmup:**
- First request: 30-120s (model loading) + 2-3s (inference)
- Subsequent: 2-3s

**With warmup:**
- First request: 2-3s (inference only)
- Subsequent: 2-3s

### Keep-Alive Behavior

Each API request resets the keep-alive timer:

```python
# Every request includes keep_alive
data = {
    "model": "qwen2.5-abliterate:14b",
    "messages": [...],
    "keep_alive": "30m"  # Timer resets
}
```

**Timeline example:**
```
00:00 - System start (warmup)
00:30 - User request 1 (timer → 30:30)
15:00 - User request 2 (timer → 45:00)
45:00 - No activity for 30m, model unloads
45:30 - Next request reloads model
```

## Testing

### Quick Test Script

```bash
# Test warmup and keep-alive
python scripts/test_llm_warmup.py

# Test with custom model
python scripts/test_llm_warmup.py "llama3.2" "15m"

# Using uv
uv run python scripts/test_llm_warmup.py
```

### Manual Testing

#### 1. Check Ollama is running
```bash
curl http://localhost:11434/api/tags
```

#### 2. Test warmup manually
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "huihui_ai/qwen2.5-abliterate:14b",
  "prompt": "",
  "keep_alive": "30m"
}'
```

#### 3. Check loaded models
```bash
curl http://localhost:11434/api/ps
```

Expected output:
```json
{
  "models": [
    {
      "name": "huihui_ai/qwen2.5-abliterate:14b",
      "size_vram": 8589934592,
      "expires_at": "2024-01-01T12:30:00Z"
    }
  ]
}
```

#### 4. Test fast follow-up
```bash
time curl http://localhost:11434/api/chat -d '{
  "model": "huihui_ai/qwen2.5-abliterate:14b",
  "messages": [{"role": "user", "content": "Hi"}],
  "stream": false,
  "keep_alive": "30m"
}'
```

Should complete in <3 seconds if model is pre-loaded.

## Monitoring

### Startup Logs

Look for these messages during GLaDOS startup:

```
INFO: LLM Warmup: Pre-loading model 'huihui_ai/qwen2.5-abliterate:14b' into Ollama memory...
SUCCESS: LLM Warmup: Model 'huihui_ai/qwen2.5-abliterate:14b' successfully loaded into memory. Will remain loaded for 30m.
```

### Error Scenarios

**Ollama not running:**
```
WARNING: LLM Warmup: Could not connect to Ollama at http://localhost:11434/api/chat.
Please ensure Ollama is running. System will continue but LLM requests may fail.
```

**Timeout (large model):**
```
WARNING: LLM Warmup: Timeout while loading model 'huihui_ai/qwen2.5-abliterate:14b'.
The model is likely large and may take longer on first real request. System will continue normally.
```

**HTTP Error:**
```
WARNING: LLM Warmup: Received HTTP 404 from Ollama.
Model may not be pre-loaded, but system will continue.
```

### Runtime Monitoring

Check Ollama logs:
```bash
# Ollama server logs
journalctl -u ollama -f

# Or if running manually
ollama serve  # Watch output
```

## Performance Metrics

### Expected Timings

| Scenario | Without Warmup | With Warmup |
|----------|---------------|-------------|
| System startup | 3-5s | 30-120s |
| First LLM request | 30-120s | 2-3s |
| Subsequent requests | 2-3s | 2-3s |
| User perceived delay | 😩 Long | 😊 Instant |

### Memory Usage

| Model Size | VRAM Usage | Recommended keep_alive |
|------------|------------|----------------------|
| 7B (e.g., Llama 3.2) | ~4GB | 30m - 1h |
| 14B (e.g., Qwen 2.5) | ~8GB | 30m |
| 20B+ | ~12GB+ | 15m - 30m |

## Troubleshooting

### Issue: Warmup always times out

**Cause**: Model is very large or system is slow

**Solution**:
1. Increase timeout in `engine.py:303`:
   ```python
   timeout=240,  # 4 minutes instead of 2
   ```
2. Or let it timeout - system works normally, just slower on first request

### Issue: Model unloads too quickly

**Cause**: `keep_alive_timeout` too short

**Solution**: Increase in config:
```yaml
keep_alive_timeout: "1h"  # Instead of 30m
```

### Issue: High memory usage

**Cause**: Model stays loaded for too long

**Solution**: Decrease timeout:
```yaml
keep_alive_timeout: "10m"  # Instead of 30m
```

### Issue: Warmup fails but system works

**Behavior**: This is normal! Warmup is non-critical

**What happens**:
- System logs warning
- Continues initialization
- First user request will trigger model load
- Subsequent requests are fast

## API Reference

### GladosConfig

```python
class GladosConfig(BaseModel):
    llm_model: str
    completion_url: HttpUrl
    keep_alive_timeout: str = "30m"  # NEW: Keep-alive timeout
    # ... other fields
```

### LanguageModelProcessor

```python
def __init__(
    self,
    # ... other parameters
    keep_alive_timeout: str = "30m",  # NEW: Keep-alive timeout
) -> None:
    self.keep_alive_timeout = keep_alive_timeout
```

### Glados._warmup_llm()

```python
def _warmup_llm(self) -> None:
    """
    Pre-load LLM model into Ollama memory.

    Non-blocking: Logs warning on failure but continues initialization.
    Uses /api/generate endpoint with empty prompt for minimal overhead.
    """
```

## Best Practices

### 1. Set Appropriate Timeout
- **Default (30m)**: Good for most users
- **Increase**: If you have frequent conversations
- **Decrease**: If memory is limited

### 2. Monitor First Run
Watch logs on first system start to verify warmup works:
```bash
uv run glados start --config configs/glados_config.yaml | grep -i warmup
```

### 3. Test After Changes
Always run test script after modifying config:
```bash
python scripts/test_llm_warmup.py
```

### 4. Consider VRAM
Check available VRAM before setting long timeouts:
```bash
# NVIDIA GPUs
nvidia-smi

# AMD GPUs
rocm-smi
```

### 5. Balance Performance vs Resources
```
Short timeout = Low memory, slower responses
Long timeout = High memory, instant responses
```

## Future Enhancements

### Planned Features
1. **Auto-tuning**: Automatically adjust `keep_alive` based on usage patterns
2. **Multi-tier LLM**: Fast small model for simple queries, large model for complex
3. **Lazy TTS/RVC**: Unload TTS/RVC when idle (see issue #TODO)
4. **Context trimming**: Auto-trim conversation history (see issue #TODO)

### Possible Improvements
1. Parallel warmup (ASR + LLM simultaneously)
2. Predictive loading based on time of day
3. Streaming warmup status to UI
4. Health check endpoint for monitoring

## Related Documentation

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [GLaDOS Architecture](./ARCHITECTURE.md)
- [Performance Optimization Guide](./PERFORMANCE.md)

## Changelog

### v1.0 (2024-10-28)
- ✨ Added automatic LLM warmup on system start
- ✨ Added `keep_alive_timeout` configuration parameter
- ✨ Added graceful error handling for warmup failures
- ✨ Created `test_llm_warmup.py` test script
- 📝 Added comprehensive documentation

---

**Questions?** Check the [FAQ](./FAQ.md) or open an issue on GitHub.
