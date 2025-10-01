from typing import List, Tuple
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
try:
    from youtube_transcript_api import TranscriptsDisabled  # v0.6.x
except Exception:
    class TranscriptsDisabled(Exception): pass
from requests.exceptions import RequestException

def parse_video_id(url: str) -> str:
    u = urlparse(url)
    if u.netloc.endswith("youtu.be"):
        vid = u.path.strip("/").split("/")[0]
        if vid: return vid
    qs = parse_qs(u.query or "")
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    if "/shorts/" in u.path:
        return u.path.split("/shorts/")[1].split("/")[0].split("?")[0]
    raise ValueError("Could not parse YouTube video ID from the URL.")

def _to_items(raw: List[dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in raw:
        secs = int(it.get("start", 0))
        h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
        out.append((f"{h:02d}:{m:02d}:{s:02d}", it.get("text", "")))
    return out

def fetch_transcript_for_video(video_id: str) -> List[Tuple[str, str]]:
    """
    Returns [(timestamp, text), ...] or [] if unavailable/blocked.
    Works with both 0.6.x and 1.x youtube-transcript-api.
    Tries manual->generated->any(+translate to en).
    """
    try:
        # Legacy 0.6.x
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                raw = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "hi"])
                return _to_items(raw)
            except NoTranscriptFound:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                t = transcripts.find_generated_transcript(["en", "en-US", "hi"])
                return _to_items(t.fetch())

        # 1.x flow
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = transcripts.find_transcript(["en", "en-US", "hi"])
            return _to_items(t.fetch())
        except NoTranscriptFound:
            pass
        try:
            t = transcripts.find_generated_transcript(["en", "en-US", "hi"])
            return _to_items(t.fetch())
        except NoTranscriptFound:
            pass

        for t in transcripts:
            try:
                t_en = t.translate("en")
                return _to_items(t_en.fetch())
            except Exception:
                return _to_items(t.fetch())

        return []

    except (NoTranscriptFound, TranscriptsDisabled, RequestException, KeyError, ValueError):
        return []
    except Exception:
        return []
