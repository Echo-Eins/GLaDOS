from __future__ import annotations

import math
from dataclasses import dataclass
from time import time as time_now
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from .config import RVCConfig

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore


def _lazy_import_parselmouth():  # pragma: no cover - imported lazily
    import importlib

    return importlib.import_module("parselmouth")


def _lazy_import_pyworld():  # pragma: no cover - imported lazily
    import importlib

    return importlib.import_module("pyworld")


def _lazy_import_torchcrepe():  # pragma: no cover - imported lazily
    import importlib

    return importlib.import_module("torchcrepe")


@dataclass(slots=True)
class RVCTiming:
    feature: float = 0.0
    f0: float = 0.0
    synthesis: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.feature, self.f0, self.synthesis)


class RVCPipeline:
    """Inference pipeline adapted from the official RVC WebUI implementation."""

    def __init__(self, target_sample_rate: int, config: RVCConfig):
        self.config = config
        self.target_sample_rate = target_sample_rate
        self.hubert_sample_rate = 16000
        self.window = 160
        self.highpass_b, self.highpass_a = signal.butter(N=5, Wn=48, btype="high", fs=self.hubert_sample_rate)

        self.t_pad = self.hubert_sample_rate * self.config.x_pad
        self.t_pad_tgt = self.target_sample_rate * self.config.x_pad
        self.t_query = self.hubert_sample_rate * self.config.x_query
        self.t_center = self.hubert_sample_rate * self.config.x_center
        self.t_max = self.hubert_sample_rate * self.config.x_max

    @staticmethod
    def _change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
        import librosa  # pragma: no cover - heavy import

        rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
        rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
        rms1_t = torch.from_numpy(rms1)
        rms2_t = torch.from_numpy(rms2)
        rms1_t = F.interpolate(rms1_t.unsqueeze(0), size=data2.shape[0], mode="linear", align_corners=False).squeeze()
        rms2_t = F.interpolate(rms2_t.unsqueeze(0), size=data2.shape[0], mode="linear", align_corners=False).squeeze()
        rms2_t = torch.maximum(rms2_t, torch.zeros_like(rms2_t) + 1e-6)
        mixed = (
            torch.pow(rms1_t, torch.tensor(1 - rate))
            * torch.pow(rms2_t, torch.tensor(rate - 1))
        ).numpy()
        return data2 * mixed

    def _compute_f0(
        self,
        audio: np.ndarray,
        p_len: int,
        f0_shift: int,
        method: str,
        inp_f0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        time_step = self.window / self.hubert_sample_rate * 1000
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        method = method.lower()

        if method == "pm":
            parselmouth = _lazy_import_parselmouth()
            f0 = (
                parselmouth.Sound(audio, self.hubert_sample_rate)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = max((p_len - len(f0) + 1) // 2, 0)
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, (pad_size, p_len - len(f0) - pad_size))
        elif method == "crepe":
            torchcrepe = _lazy_import_torchcrepe()
            audio_t = torch.tensor(np.copy(audio))[None].float()
            f0, periodicity = torchcrepe.predict(
                audio_t,
                self.hubert_sample_rate,
                self.window,
                f0_min,
                f0_max,
                "full",
                batch_size=512,
                device=self.config.device,
                return_periodicity=True,
            )
            periodicity = torchcrepe.filter.median(periodicity, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[periodicity < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        else:  # default harvest
            pyworld = _lazy_import_pyworld()
            f0, t = pyworld.harvest(audio.astype(np.double), fs=self.hubert_sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.hubert_sample_rate)

        f0 *= math.pow(2, f0_shift / 12)
        tf0 = self.hubert_sample_rate // self.window
        if inp_f0 is not None and len(inp_f0) > 0:
            delta_t = int(round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1))
            replace_f0 = np.interp(np.arange(delta_t), inp_f0[:, 0] * 100, inp_f0[:, 1])
            start = self.config.x_pad * tf0
            end = start + len(replace_f0)
            f0_slice = f0[start:end]
            shape = f0_slice.shape[0]
            f0[start : start + shape] = replace_f0[:shape]

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0

    def _run_vc(
        self,
        hubert_model: torch.nn.Module,
        net_g: torch.nn.Module,
        sid: torch.Tensor,
        audio: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        timing: RVCTiming,
        index: Optional["faiss.Index"] = None,
        big_npy: Optional[np.ndarray] = None,
        index_rate: float = 0.0,
        version: str = "v2",
    ) -> np.ndarray:
        feats = torch.from_numpy(audio)
        feats = feats.half() if self.config.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.zeros_like(feats, dtype=torch.bool, device=self.config.device)
        inputs = {
            "source": feats.to(self.config.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = time_now()
        with torch.no_grad():
            logits = hubert_model.extract_features(**inputs)
            feats = logits[0] if isinstance(logits, tuple) else logits
            if isinstance(feats, tuple):
                feats = feats[0]
        if hasattr(hubert_model, "final_proj") and version == "v1":
            feats = hubert_model.final_proj(feats)
        if self.config.is_half:
            feats = feats.half()
        if index_rate > 0 and index is not None and big_npy is not None:
            npy = feats[0].cpu().numpy()
            if self.config.is_half:
                npy = npy.astype("float32")
            score, idx = index.search(npy, k=8)
            weight = np.square(1 / np.maximum(score, 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[idx] * np.expand_dims(weight, axis=2), axis=1)
            if self.config.is_half:
                npy = npy.astype("float16")
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.config.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2, mode="linear", align_corners=False).permute(0, 2, 1)
        t1 = time_now()
        p_len = audio.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len_tensor = torch.tensor([p_len], device=self.config.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio_pred = net_g.infer(feats, p_len_tensor, pitch, pitchf, sid)[0][0, 0]
            else:
                audio_pred = net_g.infer(feats, p_len_tensor, sid)[0][0, 0]
        t2 = time_now()
        timing.feature += t1 - t0
        timing.synthesis += t2 - t1
        return audio_pred.data.cpu().float().numpy()

    def infer(
        self,
        hubert_model: torch.nn.Module,
        net_g: torch.nn.Module,
        sid: int,
        audio: np.ndarray,
        *,
        f0_shift: int,
        f0_method: str,
        index: Optional["faiss.Index"],
        index_embeddings: Optional[np.ndarray],
        index_rate: float,
        if_f0: bool,
        version: str,
        inp_f0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, RVCTiming]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = signal.filtfilt(self.highpass_b, self.highpass_a, audio)
        padded = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        optimum_points: list[int] = []
        if padded.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(padded[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                search_slice = audio_sum[t - self.t_query : t + self.t_query]
                optimum = int(np.argmin(search_slice))
                optimum_points.append(t - self.t_query + optimum)

        padded = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = padded.shape[0] // self.window
        timing = RVCTiming()
        sid_tensor = torch.tensor([sid], device=self.config.device).long()
        pitch = pitchf = None
        if if_f0:
            t_start = time_now()
            coarse, fine = self._compute_f0(padded, p_len, f0_shift, f0_method, inp_f0)
            timing.f0 += time_now() - t_start
            coarse = coarse[:p_len]
            fine = fine[:p_len]
            if "mps" not in self.config.device and "xpu" not in self.config.device:
                fine = fine.astype(np.float32)
            pitch = torch.tensor(coarse, device=self.config.device).unsqueeze(0).long()
            pitchf = torch.tensor(fine, device=self.config.device).unsqueeze(0).float()

        chunks: list[np.ndarray] = []
        cursor = 0
        for t in optimum_points:
            t = t // self.window * self.window
            start = cursor
            end = t + self.t_pad * 2 + self.window
            pitch_slice = pitch[:, start // self.window : (t + self.t_pad * 2) // self.window] if pitch is not None else None
            pitchf_slice = pitchf[:, start // self.window : (t + self.t_pad * 2) // self.window] if pitchf is not None else None
            chunk = self._run_vc(
                hubert_model,
                net_g,
                sid_tensor,
                padded[start:end],
                pitch_slice,
                pitchf_slice,
                timing,
                index,
                index_embeddings,
                index_rate,
                version,
            )
            chunks.append(chunk[self.t_pad_tgt : -self.t_pad_tgt])
            cursor = t

        pitch_slice = pitch[:, cursor // self.window :] if pitch is not None else None
        pitchf_slice = pitchf[:, cursor // self.window :] if pitchf is not None else None
        chunk = self._run_vc(
            hubert_model,
            net_g,
            sid_tensor,
            padded[cursor:],
            pitch_slice,
            pitchf_slice,
            timing,
            index,
            index_embeddings,
            index_rate,
            version,
        )
        chunks.append(chunk[self.t_pad_tgt : -self.t_pad_tgt])
        audio_out = np.concatenate(chunks)

        if self.config.rms_mix_rate != 1.0:
            audio_out = self._change_rms(audio, self.hubert_sample_rate, audio_out, self.target_sample_rate, self.config.rms_mix_rate)

        resample_target = self.config.resample_sr if self.config.resample_sr >= 16000 else self.target_sample_rate
        if self.target_sample_rate != resample_target:
            import librosa  # pragma: no cover

            audio_out = librosa.resample(audio_out, orig_sr=self.target_sample_rate, target_sr=resample_target)

        peak = np.abs(audio_out).max() / 0.99
        max_int16 = 32768 if peak <= 1 else 32768 / peak
        audio_out = (audio_out * max_int16).astype(np.int16)
        return audio_out, timing
