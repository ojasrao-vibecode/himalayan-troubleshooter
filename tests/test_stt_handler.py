from unittest.mock import patch
import numpy as np
import stt_handler


def test_empty_audio_returns_empty_string():
    result = stt_handler.transcribe(16000, np.array([]))
    assert result == ""


def test_none_audio_returns_empty_string():
    result = stt_handler.transcribe(16000, None)
    assert result == ""


def test_stereo_converted_to_mono():
    stereo = np.array([[100, 200], [300, 400]], dtype=np.float32)
    mono = stereo.mean(axis=1)
    assert mono.shape == (2,)
    assert mono[0] == 150.0


def test_int16_audio_normalised_to_float_range():
    audio = np.array([32768, -32768, 16384], dtype=np.float32)
    peak = np.abs(audio).max()
    normalised = audio / peak
    assert normalised.max() <= 1.0
    assert normalised.min() >= -1.0


def test_whisper_failure_returns_error_string():
    with patch("stt_handler._get_model") as mock_get_model:
        mock_get_model.return_value.transcribe.side_effect = RuntimeError("model error")
        result = stt_handler.transcribe(16000, np.ones(16000, dtype=np.float32))
    assert result.startswith("[Voice transcription failed:")


def test_resampling_changes_length():
    original = np.ones(44100, dtype=np.float32)
    target_len = int(len(original) * 16000 / 44100)
    resampled = np.interp(
        np.linspace(0, len(original) - 1, target_len),
        np.arange(len(original)),
        original,
    ).astype(np.float32)
    assert len(resampled) == target_len
    assert len(resampled) < len(original)
