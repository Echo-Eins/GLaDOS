"""RVC (Retrieval-based Voice Conversion) processor for GLaDOS voice."""

from __future__ import annotations

import tempfile
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from loguru import logger

from ..utils.resources import resource_path

from .rvc import (
    RVCConfig,
    RVCPipeline,
    load_hubert_model,
    resolve_hubert_checkpoint,
)
from .rvc.infer_pack import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


class RVCProcessor:
    """Full featured RVC voice conversion pipeline."""

    def __init__(
        self,
        model_path: Path | str,
        index_path: Path | str | None = None,
        *,
        device: str | None = None,
        f0_method: str = "harvest",
        f0_up_key: int = 0,
        index_rate: float = 0.75,
        hubert_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.f0_method = f0_method
        self.f0_up_key = f0_up_key
        self.index_rate = max(0.0, min(index_rate, 1.0))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.is_half = self.device.type == "cuda"

        logger.info("Initializing RVC processor on {}".format(self.device))

        self._load_model()
        self._load_hubert(hubert_path)
        self._load_index()

        config = RVCConfig(
            device=str(self.device),
            is_half=self.is_half,
        )
        config.resample_sr = self.sample_rate
        self.pipeline = RVCPipeline(self.sample_rate, config)

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"RVC model file not found: {self.model_path}")
        logger.info("Loading RVC model from %s", self.model_path)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported RVC checkpoint format")
        self.config = checkpoint.get("config", [])
        if not self.config:
            raise ValueError("Invalid RVC checkpoint: missing config")
        self.version = checkpoint.get("version", "v2")
        self.if_f0 = checkpoint.get("f0", 1) == 1
        self.sample_rate = int(self.config[-1])
        state_dict = checkpoint.get("weight") or checkpoint
        if not isinstance(state_dict, (dict, OrderedDict)):
            raise ValueError("Invalid RVC checkpoint weights")
        n_speakers = state_dict.get("emb_g.weight")
        if n_speakers is None:
            raise ValueError("RVC checkpoint does not contain speaker embedding weights")
        self.config[-3] = n_speakers.shape[0]

        model_cls: type[torch.nn.Module]
        if self.version == "v1":
            model_cls = SynthesizerTrnMs256NSFsid if self.if_f0 else SynthesizerTrnMs256NSFsid_nono
        else:
            model_cls = SynthesizerTrnMs768NSFsid if self.if_f0 else SynthesizerTrnMs768NSFsid_nono

        self.net_g = model_cls(*self.config, is_half=self.is_half)
        if hasattr(self.net_g, "enc_q"):
            delattr(self.net_g, "enc_q")
        self.net_g.load_state_dict(state_dict, strict=False)
        self.net_g.eval().to(self.device)
        if self.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        logger.success("RVC model loaded (sr=%s, f0=%s, version=%s)", self.sample_rate, self.if_f0, self.version)

    def _ensure_hubert_checkpoint(self, hubert_path: Optional[Path]) -> Path:
        if hubert_path and Path(hubert_path).exists():
            return Path(hubert_path)
        package_default = resolve_hubert_checkpoint(resource_path("models/RVC"))
        if package_default:
            return package_default
        target_dir = self.model_path.parent
        candidate = target_dir / "hubert_base.pt"
        if candidate.exists():
            return candidate
        url = (
            "https://huggingface.co/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/"
            "resolve/main/assets/hubert/hubert_base.pt?download=1"
        )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading HuBERT checkpoint to %s", candidate)
        import requests

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, dir=str(candidate.parent)) as tmp_file:
                temp_path = Path(tmp_file.name)
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        tmp_file.write(chunk)
        temp_path.replace(candidate)
        return candidate

    def _load_hubert(self, hubert_path: Path | str | None) -> None:
        checkpoint_path = self._ensure_hubert_checkpoint(Path(hubert_path) if hubert_path else None)
        logger.info("Loading HuBERT model from %s", checkpoint_path)
        self.hubert = load_hubert_model(checkpoint_path, self.device, is_half=self.is_half)
        logger.success("HuBERT model loaded")

    def _load_index(self) -> None:
        if not self.index_path or not self.index_path.exists() or not FAISS_AVAILABLE:
            if self.index_path and not FAISS_AVAILABLE:
                logger.warning("FAISS not available, skipping index loading")
            self.index = None
            self.index_embeddings = None
            return
        logger.info("Loading RVC index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
        self.index_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        logger.success("RVC index loaded")

    def process(self, audio: np.ndarray, input_sample_rate: int = 48000) -> np.ndarray:
        if audio.size == 0:
            return audio
        if input_sample_rate != 16000:
            logger.debug("Resampling audio from %sHz to 16000Hz for RVC", input_sample_rate)
            import librosa  # type: ignore

            audio = librosa.resample(audio, orig_sr=input_sample_rate, target_sr=16000).astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        converted, timing = self.pipeline.infer(
            self.hubert,
            self.net_g,
            sid=0,
            audio=audio,
            f0_shift=self.f0_up_key,
            f0_method=self.f0_method,
            index=self.index,
            index_embeddings=getattr(self, "index_embeddings", None),
            index_rate=self.index_rate,
            if_f0=self.if_f0,
            version=self.version,
        )
        logger.debug(
            "RVC timings: feature=%.3fs f0=%.3fs infer=%.3fs",
            timing.feature,
            timing.f0,
            timing.synthesis,
        )
        converted = converted.astype(np.float32) / 32768.0
        return np.clip(converted, -1.0, 1.0)


def create_rvc_processor(
    model_path: Path | str,
    index_path: Path | str | None = None,
    simple_mode: bool = False,  # Retained for backward compatibility
    **kwargs: object,
) -> RVCProcessor:
    """Factory to create an :class:`RVCProcessor`."""

    return RVCProcessor(model_path=model_path, index_path=index_path, **kwargs)
