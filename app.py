
# app.py — Multi-modal YouTube/Upload QA with text + visual evidence
# - Text embeddings: Google text-embedding-004
# - Visual embeddings: CLIP (sentence-transformers "clip-ViT-B-32")
# - Transcription fallback: AssemblyAI
# - Screenshot preview for each retrieved chunk timestamp
# - Enhanced: Handles silent videos with image-only storytelling

from __future__ import annotations

import os, math, socket, mimetypes, re
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
import requests.packages.urllib3.util.connection as urllib3_cn
import google.generativeai as genai
from PIL import Image
import numpy as np
import moviepy.editor as mp

# ---- Load .env FIRST so service modules can see keys ----
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# ---- Prefer IPv4 (reduces odd network issues) ----
def _force_ipv4_only():
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
_force_ipv4_only()

# ---- Project services ----
from services.youtube import parse_video_id, fetch_transcript_for_video
from services.assembly_ai import transcribe_with_assemblyai
from services.video import try_download_youtube_mp4, extract_frames

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing in .env")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="YouTube Chat (Gemini)", page_icon="📻", layout="wide")

st.markdown("""
<style>
.app-title { font-size: 28px; font-weight: 700; margin-bottom: 2px; }
.app-sub   { color:#666; margin-bottom: 18px; }
.chunk-badge { background:#eef2ff; color:#334155; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">📻 Chat with YouTube Video</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Use YouTube captions when available. Otherwise upload audio/video and we’ll transcribe with AssemblyAI. Supports screenshots as visual evidence.</div>', unsafe_allow_html=True)

# ====================== Helpers =====================
def has_audio(file_path: str) -> bool:
    try:
        clip = mp.VideoFileClip(file_path)
        return clip.audio is not None
    except Exception:
        return False

# ================= Embedding =====================
def _extract_embedding(resp):
    if not resp:
        raise ValueError("Empty embedding response")
    if isinstance(resp, dict):
        if "embedding" in resp:
            emb = resp["embedding"]
            return emb.get("values") if isinstance(emb, dict) else emb
        if "embeddings" in resp:
            e = resp["embeddings"][0]
            return e.get("values") or e.get("embedding") or e
    if hasattr(resp, "embedding"):
        e = resp.embedding
        return getattr(e, "values", e)
    if isinstance(resp, list) and resp:
        e0 = resp[0]
        return e0.get("values") or e0.get("embedding") or e0
    raise ValueError(f"Unexpected embedding response type: {type(resp)}")

def embed_one(text: str) -> List[float]:
    if not text.strip(): return []
    try:
        r = genai.embed_content(model="models/text-embedding-004", content=text)
    except:
        r = genai.embed_content(model="text-embedding-004", content=text)
    return _extract_embedding(r)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return [embed_one(t) for t in texts if t.strip()]

@st.cache_resource(show_spinner=False)
def get_clip_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("clip-ViT-B-32")

def embed_image_file(path: str) -> List[float]:
    model = get_clip_model()
    img = Image.open(path).convert("RGB")
    vec = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32).tolist()

def embed_text_mm(text: str) -> List[float]:
    model = get_clip_model()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()

# =================== Chat Model ====================
def pick_chat_model():
    try:
        # Fetch all models
        models = genai.list_models()
        
        # Filter to generative chat models only
        available = [
            m.name for m in models
            if "generateContent" in m.supported_generation_methods
        ]
        
        # Sort by latest model name (e.g. gemini-1.5-pro > gemini-pro)
        if available:
            latest = sorted(available, reverse=True)[0]
            return genai.GenerativeModel(model_name=latest)
        else:
            raise RuntimeError("No supported Gemini model found in your API access.")
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Gemini models: {e}")

# =================== Retrieval Utils ====================
def chunk_transcript(items: List[Tuple[str, str]], max_chars=600):
    chunks, buf, anchor, count = [], [], None, 0
    for ts, txt in items:
        if anchor is None: anchor = ts
        if count + len(txt) + 1 > max_chars and buf:
            chunks.append((anchor, " ".join(buf).strip()))
            buf, anchor, count = [], ts, 0
        buf.append(txt); count += len(txt) + 1
    if buf:
        chunks.append((anchor, " ".join(buf).strip()))
    return chunks

def cosine(u: List[float], v: List[float]) -> float:
    dot = sum(a*b for a,b in zip(u,v))
    nu = math.sqrt(sum(a*a for a in u)) + 1e-8
    nv = math.sqrt(sum(b*b for b in v)) + 1e-8
    return dot/(nu*nv)

def retrieve(qv: List[float], embs: List[List[float]], top_k: int = 4):
    return [i for i,_ in sorted(((i,cosine(qv,e)) for i,e in enumerate(embs) if e), key=lambda x:x[1], reverse=True)[:top_k]]

_TS_RE = re.compile(r"^(?:(\d+):)?([0-5]?\d):([0-5]?\d)$")
def ts_to_seconds(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m: return 0.0
    h, m_, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    return h*3600 + m_*60 + s

def nearest_frame_for_ts(ts: str, frames: list[tuple[str,str]]) -> tuple[str,str] | None:
    target = ts_to_seconds(ts)
    return min(frames, key=lambda x: abs(ts_to_seconds(x[0]) - target), default=None)

def build_visual_index_from_video(video_path: str, every_sec=2.0):
    with st.spinner("Extracting key frames and computing image embeddings..."):
        frames = extract_frames(video_path, outdir="frames", every_sec=every_sec, max_frames=240)
        embs = [embed_image_file(p) for _,p in frames]
        st.session_state.vis_frames = frames
        st.session_state.vis_embs = embs

# =================== App State ====================
for key in ["idx_chunks", "idx_embs", "vis_frames", "vis_embs", "source"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "source" else ""

# =================== Tabs: YouTube / Upload ====================
yt_tab, upload_tab = st.tabs(["YouTube link", "Upload (AssemblyAI)"])

with yt_tab:
    yt_url = st.text_input("YouTube URL")
    if st.button("Fetch captions") and yt_url:
        try:
            with st.spinner("Fetching YouTube transcript and embedding..."):
                vid = parse_video_id(yt_url)
                items = fetch_transcript_for_video(vid)
                chunks = chunk_transcript(items)
                texts = [t for _,t in chunks]
                embs = embed_texts(texts)
                st.session_state.idx_chunks = chunks
                st.session_state.idx_embs = embs
                st.session_state.source = f"YouTube:{vid}"
        except Exception as e:
            st.error(str(e))
    if st.button("Index visuals"):
        path = try_download_youtube_mp4(yt_url)
        if path:
            build_visual_index_from_video(path)

with upload_tab:
    file = st.file_uploader("Choose audio/video file")
    if st.button("Transcribe") and file:
        path = f"uploads/{file.name}"
        os.makedirs("uploads", exist_ok=True)
        with open(path, "wb") as f: f.write(file.read())
        if has_audio(path):
            with st.spinner("Transcribing with AssemblyAI..."):
                items = transcribe_with_assemblyai(path)
                chunks = chunk_transcript(items)
                texts = [t for _,t in chunks]
                embs = embed_texts(texts)
                st.session_state.idx_chunks = chunks
                st.session_state.idx_embs = embs
                st.session_state.source = f"Upload:{file.name}"
        else:
            st.warning("No audio found. Using image-only analysis.")
            st.session_state.idx_chunks = []
            st.session_state.idx_embs = []
            st.session_state.source = f"Upload:{file.name} (silent)"
        build_visual_index_from_video(path)

# =================== QA Interface ====================
st.divider()
st.subheader("Ask a question")
q = st.text_input("Your question")

if q and (st.session_state.idx_embs or st.session_state.vis_embs):
    top_idx, ctx = [], ""
    if st.session_state.idx_embs:
        qv = embed_one(q)
        top_idx = retrieve(qv, st.session_state.idx_embs)
        ctx = "\n\n".join([f"[Chunk {i+1} @ {st.session_state.idx_chunks[i][0]}] {st.session_state.idx_chunks[i][1]}" for i in top_idx])

    vis_hits = []
    if st.session_state.vis_embs:
        try:
            qv_vis = embed_text_mm(q)
            ids = retrieve(qv_vis, st.session_state.vis_embs)
            vis_hits = [st.session_state.vis_frames[i] for i in ids]
        except:
            pass
    for i in top_idx:
        ts = st.session_state.idx_chunks[i][0]
        frame = nearest_frame_for_ts(ts, st.session_state.vis_frames)
        if frame and frame not in vis_hits:
            vis_hits.append(frame)

    if vis_hits:
        st.caption("Visual evidence:")
        cols = st.columns(len(vis_hits))
        for col, (ts, img) in zip(cols, vis_hits):
            col.image(img, caption=f"Visual Frame @ {ts}", use_container_width=True)

    # Prompt setup
    if not ctx and vis_hits:
        sys_prompt = "Describe in detail what is happening in the visual frames, assuming this is a silent video. Create a storyline based on image sequence."
    else:
        sys_prompt = "Answer using ONLY the provided transcript chunks and optional visual frames. Be specific."

    parts = [f"System instruction: {sys_prompt}"]
    if ctx: parts.append(f"Context:\n{ctx}")
    for ts, img in vis_hits[:2]:
        with open(img, "rb") as f:
            data = f.read()
        mime, _ = mimetypes.guess_type(img)
        parts.append({"mime_type": mime or "image/jpeg", "data": data})
    parts.append(f"Question: {q}")

    model = pick_chat_model()
    try:
        chat = model.start_chat(history=[])
        resp = chat.send_message(parts)
        st.write(resp.text.strip())
    except Exception as e:
        st.error(f"Gemini response error: {e}")

    st.markdown("**Evidence**  ")
    st.markdown(f"Text: {', '.join(f'Chunk {i+1} @ {st.session_state.idx_chunks[i][0]}' for i in top_idx) or 'None'}")
    st.markdown(f"Visual: {', '.join(f'Visual Frame @ {ts}' for ts,_ in vis_hits) or 'None'}")

elif q:
    st.info("Please index the video first using YouTube link or Upload.")
