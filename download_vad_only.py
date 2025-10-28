#!/usr/bin/env python3
"""Download only the VAD model required for voice detection."""

import asyncio
from hashlib import sha256
from pathlib import Path
import sys

import httpx
from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn
from rich import print as rprint


VAD_MODEL = {
    "path": "models/ASR/silero_vad_v5.onnx",
    "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad_v5.onnx",
    "checksum": "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396",
}


async def download_with_progress(
    client: httpx.AsyncClient,
    url: str,
    file_path: Path,
    expected_checksum: str,
    progress: Progress,
) -> bool:
    """Download a file with progress tracking and checksum verification."""
    task_id = progress.add_task(f"Downloading {file_path.name}", status="")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    hash_sha256 = sha256()

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            if total_size:
                progress.update(task_id, total=total_size)

            with file_path.open(mode="wb") as f:
                async for chunk in response.aiter_bytes(32768):
                    f.write(chunk)
                    hash_sha256.update(chunk)
                    progress.advance(task_id, len(chunk))

        # Verify checksum
        actual_checksum = hash_sha256.hexdigest()
        if actual_checksum != expected_checksum:
            progress.update(task_id, status="[bold red]Checksum failed")
            file_path.unlink()
            return False
        else:
            progress.update(task_id, status="[bold green]OK")
            return True

    except Exception as e:
        progress.update(task_id, status=f"[bold red]Error: {str(e)}")
        return False


async def main() -> int:
    """Download the VAD model."""
    file_path = Path(VAD_MODEL["path"])

    # Check if already exists and valid
    if file_path.exists():
        if sha256(file_path.read_bytes()).hexdigest() == VAD_MODEL["checksum"]:
            rprint(f"[bold green]✓ {file_path.name} already exists and is valid")
            return 0
        else:
            rprint(f"[yellow]! {file_path.name} exists but checksum is invalid, re-downloading...")

    with Progress(
        TextColumn("[grey50][progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("  {task.fields[status]}"),
    ) as progress:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            success = await download_with_progress(
                client,
                VAD_MODEL["url"],
                file_path,
                VAD_MODEL["checksum"],
                progress
            )

    if success:
        rprint("\n[bold green]✓ VAD model downloaded successfully!")
        rprint("[dim]You can now run: uv run glados start --config configs/glados_ru_config.yaml")
        return 0
    else:
        rprint("\n[bold red]✗ Failed to download VAD model")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
