import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        import whisper
        # "tiny" model (~39 MB download, no ffmpeg needed for numpy input)
        _model = whisper.load_model("tiny")
    return _model


def transcribe(sample_rate: int, audio_data: np.ndarray) -> str:
    """Transcribe mic audio (numpy array from Gradio) using local Whisper."""
    if audio_data is None or audio_data.size == 0:
        return ""

    # Flatten stereo to mono
    audio = np.array(audio_data)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Normalise to float32 in [-1, 1]
    audio = audio.astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / 32768.0  # int16 range → float32

    # Resample to 16 kHz (Whisper's required sample rate)
    if sample_rate != 16000:
        target_len = int(len(audio) * 16000 / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    try:
        model = _get_model()
        result = model.transcribe(audio, fp16=False, language="en")
        return result["text"].strip()
    except Exception as exc:
        return f"[Voice transcription failed: {exc}]"
