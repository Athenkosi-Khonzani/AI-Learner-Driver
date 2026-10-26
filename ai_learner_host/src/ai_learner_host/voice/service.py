from __future__ import annotations

import asyncio
import io
import math
import os
import time
import wave

import numpy as np

from ai_learner_host.config import AppConfig


class VoiceInputService:
    """Push-to-talk style microphone capture followed by OpenAI transcription.

    Recording stops automatically after speech followed by silence, or at the
    configured maximum duration. Audio capture runs off the Qt/asyncio thread.
    """

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    MAX_SECONDS = 12.0
    INITIAL_SPEECH_TIMEOUT_SECONDS = 5.0
    SILENCE_AFTER_SPEECH_SECONDS = 1.1
    RMS_SPEECH_THRESHOLD = 0.018

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def list_input_devices() -> list[str]:
        try:
            import sounddevice as sd
        except ImportError:
            return []
        names: list[str] = []
        try:
            for device in sd.query_devices():
                if int(device.get("max_input_channels", 0)) > 0:
                    name = str(device.get("name", "")).strip()
                    if name and name not in names:
                        names.append(name)
        except Exception:
            return []
        return names

    def readiness(self) -> tuple[bool, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is required for voice transcription"
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            return False, "Install the 'sounddevice' package for microphone capture"
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "Install the 'openai' package for transcription"
        return True, "Voice input ready"

    async def transcribe_once(self) -> str:
        ready, message = self.readiness()
        if not ready:
            raise RuntimeError(message)

        pcm = await asyncio.to_thread(self._record_until_silence_sync)
        wav_bytes = self._pcm_to_wav_bytes(pcm)
        transcript = await asyncio.to_thread(self._transcribe_sync, wav_bytes)
        text = transcript.strip()
        if not text:
            raise RuntimeError("No speech was transcribed")
        return text

    def _resolve_device(self, sd):
        configured = (self.config.microphone_name or "").strip()
        if not configured or configured == "Default system microphone":
            return None
        devices = sd.query_devices()
        configured_lower = configured.lower()
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            name = str(device.get("name", ""))
            if name == configured or configured_lower in name.lower():
                return index
        raise RuntimeError(f"Configured microphone was not found: {configured}")

    def _record_until_silence_sync(self) -> np.ndarray:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install the 'sounddevice' package for microphone capture") from exc

        device = self._resolve_device(sd)
        chunks: list[np.ndarray] = []
        started_at = time.monotonic()
        last_voice_at: float | None = None
        speech_started = False

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, last_voice_at
            _ = (frames, time_info)
            if status:
                # Do not abort for benign PortAudio status flags; the final
                # transcription is still often usable.
                pass
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            chunks.append(mono)
            now = time.monotonic()
            rms = float(math.sqrt(float(np.mean(np.square(mono))) + 1e-12))
            if rms >= self.RMS_SPEECH_THRESHOLD:
                speech_started = True
                last_voice_at = now

            elapsed = now - started_at
            if elapsed >= self.MAX_SECONDS:
                raise sd.CallbackStop()
            if not speech_started and elapsed >= self.INITIAL_SPEECH_TIMEOUT_SECONDS:
                raise sd.CallbackStop()
            if (
                speech_started
                and last_voice_at is not None
                and now - last_voice_at >= self.SILENCE_AFTER_SPEECH_SECONDS
            ):
                raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            device=device,
            callback=callback,
            blocksize=1024,
        ) as stream:
            while stream.active:
                sd.sleep(50)

        if not chunks or not speech_started:
            raise RuntimeError("No speech detected. Try again and speak closer to the microphone.")

        pcm = np.concatenate(chunks)
        # Remove a small amount of trailing silence while keeping enough context.
        max_samples = int(self.MAX_SECONDS * self.SAMPLE_RATE)
        return np.clip(pcm[:max_samples], -1.0, 1.0)

    @classmethod
    def _pcm_to_wav_bytes(cls, pcm: np.ndarray) -> bytes:
        pcm16 = (np.asarray(pcm, dtype=np.float32) * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(cls.CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(cls.SAMPLE_RATE)
            wav.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    def _transcribe_sync(self, wav_bytes: bytes) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package for voice transcription") from exc

        audio = io.BytesIO(wav_bytes)
        audio.name = "rover_instruction.wav"
        client = OpenAI()
        result = client.audio.transcriptions.create(
            model=self.config.voice_transcription_model,
            file=audio,
            prompt=(
                "A short spoken driving instruction for an AI learner rover. "
                "Possible terms include rover, forward, backward, reverse, turn, "
                "left, right, red block, green block, blue block, yellow block, "
                "millimetres, centimetres, degrees, coordinate, obstacle and stop."
            ),
        )
        return str(getattr(result, "text", ""))
