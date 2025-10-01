from __future__ import annotations
import os
from typing import List, Tuple
from pathlib import Path

def _ms_to_hhmmss(ms: int | float) -> str:
    secs = int(round(float(ms) / 1000.0))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def transcribe_with_assemblyai(audio_path: str) -> List[Tuple[str, str]]:
    # Ensure .env is loaded even if caller forgot
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
    except Exception:
        pass

    import assemblyai as aai

    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY missing in .env")
    aai.settings.api_key = key

    transcriber = aai.Transcriber()
    cfg = aai.TranscriptionConfig(
        punctuate=True,
        format_text=True,
        speech_model=aai.SpeechModel.best,
        auto_highlights=False,
        speaker_labels=False,
        disfluencies=False,
    )
    t = transcriber.transcribe(audio_path, config=cfg)
    if t.status != aai.TranscriptStatus.completed:
        raise RuntimeError(f"Transcription failed: {t.status} {t.error}")

    result: List[Tuple[str, str]] = []

    # Prefer sentences; fall back to words; then whole text
    sents = getattr(t, "sentences", None)
    words = getattr(t, "words", None)

    if sents:
        for s in sents:
            result.append((_ms_to_hhmmss(s.start), s.text.strip()))
    elif words:
        chunk, start = [], None
        for w in words:
            if start is None:
                start = w.start
            chunk.append(w.text)
            if (w.end - start) >= 10_000:  # ~10s chunks
                result.append((_ms_to_hhmmss(start), " ".join(chunk)))
                chunk, start = [], None
        if chunk:
            result.append((_ms_to_hhmmss(start or 0), " ".join(chunk)))
    else:
        result.append(("00:00:00", (t.text or "").strip()))

    return result
