from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from ai_learner_host.config import AppConfig
from ai_learner_host.voice.service import VoiceInputService


class VoiceServiceTests(unittest.TestCase):
    def test_pcm_is_encoded_as_16khz_mono_wav(self):
        pcm = np.zeros(1600, dtype=np.float32)
        payload = VoiceInputService._pcm_to_wav_bytes(pcm)
        with wave.open(io.BytesIO(payload), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)

    def test_default_transcription_model(self):
        self.assertEqual(AppConfig().voice_transcription_model, "gpt-transcribe")


if __name__ == "__main__":
    unittest.main()
