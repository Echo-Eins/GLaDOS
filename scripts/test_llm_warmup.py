#!/usr/bin/env python3
"""
Test script for LLM warmup and keep_alive functionality.

This script tests:
1. LLM warmup mechanism with /api/generate
2. keep_alive parameter in API requests
3. Ollama model loading and unloading behavior

Usage:
    python scripts/test_llm_warmup.py
    # or
    uv run python scripts/test_llm_warmup.py
"""

import time
import requests
from datetime import datetime


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def test_warmup(model: str = "huihui_ai/qwen2.5-abliterate:14b", keep_alive: str = "30m"):
    """Test LLM warmup functionality."""
    base_url = "http://localhost:11434"

    log("=" * 80)
    log("Testing LLM Warmup and Keep-Alive Mechanism")
    log("=" * 80)

    # Test 1: Check if Ollama is running
    log("\n[TEST 1] Checking Ollama connection...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            log(f"✅ Ollama is running. Available models: {len(models)}")
            for m in models:
                log(f"   - {m['name']}")
        else:
            log(f"❌ Ollama returned status {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        log("❌ Cannot connect to Ollama. Please start Ollama first:")
        log("   ollama serve")
        return
    except Exception as e:
        log(f"❌ Error checking Ollama: {e}")
        return

    # Test 2: Warmup with /api/chat (with personality preprompt)
    log(f"\n[TEST 2] Testing full warmup with model '{model}'...")
    log(f"   keep_alive: {keep_alive}")
    log(f"   This simulates GLaDOS warmup: model loading + system prompt processing")

    # Minimal system prompt for testing
    warmup_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": ""},  # Empty user message
    ]

    warmup_payload = {
        "model": model,
        "messages": warmup_messages,
        "stream": False,
        "keep_alive": keep_alive,
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=warmup_payload,
            timeout=180,  # 3 minutes for full warmup
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            response_content = result.get("message", {}).get("content", "")
            log(f"✅ Warmup successful in {elapsed:.2f}s")
            log(f"   Model '{model}' is now loaded in memory")
            log(f"   System prompt processed and cached")
            log(f"   Will remain loaded for {keep_alive}")

            if response_content:
                log(f"   ⚠️  Model generated response ({len(response_content)} chars): '{response_content[:50]}'")
                log(f"   Note: GLaDOS will consume and discard this response safely")
            else:
                log(f"   ✅ Model generated empty response (ideal)")
        else:
            log(f"❌ Warmup failed with status {response.status_code}")
            log(f"   Response: {response.text[:200]}")
            return

    except requests.exceptions.Timeout:
        log(f"⚠️  Warmup timed out after 180s")
        log(f"   This is normal for large models on first load")
        log(f"   The model should be ready for subsequent requests")
    except Exception as e:
        log(f"❌ Warmup error: {e}")
        return

    # Test 3: Quick follow-up request
    log(f"\n[TEST 3] Testing fast follow-up request (model should be in memory)...")

    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'hi'"}],
        "stream": False,
        "keep_alive": keep_alive,
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=chat_payload,
            timeout=30,
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            log(f"✅ Chat request successful in {elapsed:.2f}s")
            log(f"   Response: {response.json()['message']['content'][:50]}")

            if elapsed < 3.0:
                log(f"   🎯 FAST! Model was already loaded in memory")
            else:
                log(f"   ⚠️  Slower than expected. Model may have been reloaded")
        else:
            log(f"❌ Chat request failed with status {response.status_code}")

    except Exception as e:
        log(f"❌ Chat request error: {e}")

    # Test 4: Check model status
    log(f"\n[TEST 4] Checking loaded models...")
    try:
        response = requests.get(f"{base_url}/api/ps", timeout=5)
        if response.status_code == 200:
            models_running = response.json().get("models", [])
            if models_running:
                log(f"✅ {len(models_running)} model(s) currently loaded:")
                for m in models_running:
                    log(f"   - {m['name']}")
                    if "expires_at" in m:
                        log(f"     Expires: {m['expires_at']}")
                    if "size_vram" in m:
                        vram_gb = m["size_vram"] / (1024**3)
                        log(f"     VRAM: {vram_gb:.2f} GB")
            else:
                log("⚠️  No models currently loaded")
        else:
            log(f"⚠️  Could not check model status (HTTP {response.status_code})")
    except Exception as e:
        log(f"⚠️  Could not check model status: {e}")

    # Summary
    log("\n" + "=" * 80)
    log("Test Summary:")
    log("  ✅ If warmup was fast (<5s) and chat was fast (<3s), keep_alive is working!")
    log("  ⚠️  If warmup was slow (>30s), this is normal for first load of large models")
    log("  🎯 The key metric is: chat request should be <3s if model is pre-loaded")
    log("=" * 80)


if __name__ == "__main__":
    import sys

    # Allow custom model and keep_alive from command line
    model = sys.argv[1] if len(sys.argv) > 1 else "huihui_ai/qwen2.5-abliterate:14b"
    keep_alive = sys.argv[2] if len(sys.argv) > 2 else "30m"

    test_warmup(model, keep_alive)
