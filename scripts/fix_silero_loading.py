"""Quick fix script to patch Silero TTS loading issue.

Run this if you're experiencing 'NoneType' object has no attribute 'apply_tts' error.
"""

import sys
from pathlib import Path

# Find the tts_silero_ru.py file
glados_root = Path(__file__).parent.parent
tts_file = glados_root / "src" / "glados" / "TTS" / "tts_silero_ru.py"

if not tts_file.exists():
    print(f"Error: Could not find {tts_file}")
    sys.exit(1)

print(f"Patching {tts_file}...")

# Read the file
content = tts_file.read_text(encoding='utf-8')

# Check if already patched
if "model, example_text = torch.hub.load" in content:
    print("✓ File already patched!")
    sys.exit(0)

# Apply the patch
old_code = """        try:
            # Load Silero V4 Russian model
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru'
            )

            self.model = self.model.to(self.device)"""

new_code = """        try:
            # Load Silero V4 Russian model
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru'
            )

            self.model = model.to(self.device)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    tts_file.write_text(content, encoding='utf-8')
    print("✓ Patch applied successfully!")
    print("\nNow run:")
    print("  uv run glados start --config configs/glados_ru_config.yaml")
else:
    print("✗ Could not find code to patch. File may already be fixed or has changed.")
    print("\nTry running:")
    print("  git pull origin claude/glados-audio-processing-011CUVWivYE1NCbQBCkWTYSm")
