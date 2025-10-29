"""
Chat Logger for GLaDOS conversation history.

Logs all conversations to files with format: YYMMDD(HHMMSS)_YYMMDD(HHMMSS){k|c}.log
- k: keepalive timeout
- c: manual close

Includes:
- Full conversation history (system prompt, user messages, assistant responses)
- TTS paragraph breakdown with sequence numbers
- Real timestamps for each event
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger


class ChatLogger:
    """
    Thread-safe chat logger that records conversation history to files.

    Logs are saved when:
    - Session ends manually (user closes)
    - Keepalive timeout expires

    File naming: YYMMDD(HHMMSS)_YYMMDD(HHMMSS){k|c}.log
    Example: 291025(150030)_291025(160210)k.log
    """

    def __init__(self, log_dir: str | Path = "chat_logs") -> None:
        """
        Initialize ChatLogger.

        Args:
            log_dir: Directory to store chat logs (created if doesn't exist)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.session_start_time: datetime | None = None
        self.session_messages: list[dict] = []  # Cache for session messages
        self.lock = threading.Lock()  # Thread-safe access

        logger.info(f"ChatLogger initialized: logs will be saved to {self.log_dir.absolute()}")

    def start_session(self, system_prompt: list[dict[str, str]]) -> None:
        """
        Start a new chat session.

        Args:
            system_prompt: List of personality preprompt messages (system, user, assistant)
        """
        with self.lock:
            self.session_start_time = datetime.now()
            self.session_messages = []

            # Log system prompt
            for message in system_prompt:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                self._log_message_internal(role, content)

            logger.info(f"Chat session started at {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_message(self, role: str, content: str) -> None:
        """
        Log a conversation message (user or assistant).

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        with self.lock:
            self._log_message_internal(role, content)

    def _log_message_internal(self, role: str, content: str) -> None:
        """Internal method to log message (assumes lock is held)."""
        timestamp = datetime.now()
        self.session_messages.append({
            "type": "message",
            "timestamp": timestamp,
            "role": role,
            "content": content
        })

    def log_tts_paragraph(self, paragraph_text: str, sequence_num: int) -> None:
        """
        Log a TTS paragraph (after parsing).

        Args:
            paragraph_text: Cleaned paragraph text sent to TTS
            sequence_num: Paragraph sequence number
        """
        with self.lock:
            timestamp = datetime.now()
            self.session_messages.append({
                "type": "tts_paragraph",
                "timestamp": timestamp,
                "sequence": sequence_num,
                "content": paragraph_text
            })
            logger.trace(f"ChatLogger: TTS paragraph #{sequence_num} logged")

    def end_session(self, reason: Literal["manual", "keepalive_expired"] = "manual") -> None:
        """
        End the current chat session and save to file.

        Args:
            reason: Reason for session end ("manual" or "keepalive_expired")
        """
        with self.lock:
            if not self.session_start_time or not self.session_messages:
                logger.debug("ChatLogger: No active session to end")
                return

            end_time = datetime.now()

            # Generate filename: YYMMDD(HHMMSS)_YYMMDD(HHMMSS){k|c}.log
            start_str = self.session_start_time.strftime("%y%m%d(%H%M%S)")
            end_str = end_time.strftime("%y%m%d(%H%M%S)")
            suffix = "k" if reason == "keepalive_expired" else "c"
            filename = f"{start_str}_{end_str}{suffix}.log"

            filepath = self.log_dir / filename

            try:
                with filepath.open("w", encoding="utf-8") as f:
                    # Write header
                    f.write(f"GLaDOS Chat Log\n")
                    f.write(f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Session End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"End Reason:    {reason}\n")
                    f.write("=" * 80 + "\n\n")

                    # Write all messages
                    for entry in self.session_messages:
                        timestamp_str = entry["timestamp"].strftime("%H:%M:%S")

                        if entry["type"] == "message":
                            role = entry["role"].upper()
                            content = entry["content"]
                            f.write(f"[{timestamp_str}] {role}:\n")
                            # Indent content for readability
                            for line in content.split("\n"):
                                f.write(f"  {line}\n")
                            f.write("\n")

                        elif entry["type"] == "tts_paragraph":
                            seq = entry["sequence"]
                            content = entry["content"]
                            f.write(f"[{timestamp_str}] TTS Paragraph #{seq}:\n")
                            f.write(f"  {content}\n\n")

                logger.success(f"Chat log saved: {filename} ({len(self.session_messages)} entries)")

            except Exception as e:
                logger.error(f"Failed to save chat log: {e}")

            # Clear session data
            self.session_messages = []
            self.session_start_time = None
