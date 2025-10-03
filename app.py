# app.py — Multi-modal YouTube/Upload QA with text + visual evidence
# - Text embeddings: Google text-embedding-004
# - Visual embeddings: CLIP (sentence-transformers "clip-ViT-B-32")
# - Transcription fallback: AssemblyAI
# - Screenshot preview for each retrieved chunk timestamp

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

# ---- Configure Gemini ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing in .env")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- UI --------------------
st.set_page_config(page_title="YouTube Chat (Gemini)", page_icon="📺", layout="wide")

st.markdown("""
<style>
.app-title { font-size: 28px; font-weight: 700; margin-bottom: 2px; }
.app-sub   { color:#666; margin-bottom: 18px; }
.chunk-badge { background:#eef2ff; color:#334155; padding:2px 8px; border-radius:999px; font-size:12px; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">📺 Chat with YouTube Video</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Use YouTube captions when available. Otherwise upload audio/video and we’ll transcribe with AssemblyAI. Supports screenshots as visual evidence.</div>', unsafe_allow_html=True)

# ====================== Embedding helpers =====================
def _extract_embedding(resp):
    if resp is None:
        raise ValueError("Empty embedding response")
    if isinstance(resp, dict):
        if "embedding" in resp:
            emb = resp["embedding"]
            if isinstance(emb, dict) and "values" in emb:
                return emb["values"]
            return emb
        if "embeddings" in resp and resp["embeddings"]:
            emb0 = resp["embeddings"][0]
            if isinstance(emb0, dict) and "values" in emb0:
                return emb0["values"]
            if isinstance(emb0, dict) and "embedding" in emb0:
                return emb0["embedding"]
            return emb0
    if hasattr(resp, "embedding"):
        e = resp.embedding
        if hasattr(e, "values"):
            return e.values
        return e
    if isinstance(resp, list) and resp:
        e0 = resp[0]
        if isinstance(e0, dict):
            return e0.get("values") or e0.get("embedding") or e0
        return e0
    raise ValueError(f"Unexpected embedding response type: {type(resp)}")

def embed_one(text: str) -> List[float]:
    """Safe embed one string (skip empty)."""
    if not text or not text.strip():
        return []  # skip empty text
    try:
        r = genai.embed_content(model="models/text-embedding-004", content=text)
    except Exception:
        r = genai.embed_content(model="text-embedding-004", content=text)
    return _extract_embedding(r)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Filter empty strings before embedding."""
    clean = [t for t in texts if t and t.strip()]
    return [embed_one(t) for t in clean]

# ========================== Chat model selection ===================================
def _list_chat_models() -> set[str]:
    names = set()
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or getattr(m, "generation_methods", [])
        if methods and ("generateContent" in methods or "generate_content" in methods):
            names.add(m.name)
    return names

def pick_chat_model():
    preference = [
        "models/gemini-1.5-flash","gemini-1.5-flash","models/gemini-1.5-flash-8b","gemini-1.5-flash-8b",
        "models/gemini-1.0-pro","gemini-1.0-pro"]
    available = _list_chat_models()
    for cand in preference:
        if cand in available:
            return genai.GenerativeModel(cand)
    if available:
        return genai.GenerativeModel(sorted(available)[0])
    raise RuntimeError("No Gemini chat models available for this API key.")

# ============================== Visual embeddings (CLIP) ============================
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

# ========================== Classic RAG utilities ==================================
def chunk_transcript(items: List[Tuple[str, str]], max_chars: int = 600) -> List[Tuple[str, str]]:
    chunks, buf, anchor, count = [], [], None, 0
    for ts, txt in items:
        if anchor is None:
            anchor = ts
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
    scored = sorted(((i,cosine(qv,e)) for i,e in enumerate(embs) if e), key=lambda x:x[1], reverse=True)
    return [i for i,_ in scored[:top_k]]

# -------------- timestamp helpers + frame matching ---------------------------------
_TS_RE = re.compile(r"^(?:(\d+):)?([0-5]?\d):([0-5]?\d)$")

def ts_to_seconds(ts: str) -> float:
    ts = ts.strip()
    m = _TS_RE.match(ts)
    if not m: return 0.0
    h = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    return float(h*3600+mm*60+ss)

def nearest_frame_for_ts(ts: str, frames: list[tuple[str,str]]) -> tuple[str,str] | None:
    if not frames: return None
    target = ts_to_seconds(ts)
    best, best_delta = None, 1e9
    for fts,p in frames:
        delta = abs(ts_to_seconds(fts)-target)
        if delta<best_delta:
            best_delta=delta; best=(fts,p)
    return best

# =========================== Visual index builder ===================================
def build_visual_index_from_video(video_path: str, every_sec: float = 2.0):
    frames = extract_frames(video_path, outdir="frames", every_sec=every_sec, max_frames=240)
    if not frames:
        st.warning("Could not extract frames from the video."); return
    embs = []
    with st.spinner(f"Embedding {len(frames)} frames with CLIP..."):
        for _,p in frames: embs.append(embed_image_file(p))
    st.session_state.vis_frames=frames
    st.session_state.vis_embs=embs
    st.toast(f"Indexed {len(frames)} visual frames.",icon="✅")

# (rest of your code stays exactly as you wrote, no changes needed below this point)


# =============================== App state =========================================
if "idx_chunks" not in st.session_state:
    st.session_state.idx_chunks = []          # [(ts, text)]
    st.session_state.idx_embs = []            # [vector]
    st.session_state.source = ""

if "vis_frames" not in st.session_state:
    st.session_state.vis_frames = []          # [(ts, img_path)]
    st.session_state.vis_embs = []            # [vector]

# ================================== UI Tabs ========================================
yt_tab, upload_tab = st.tabs(["🔗 YouTube link", "📤 Upload (AssemblyAI)"])

with yt_tab:
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    cols = st.columns([1,1,2,2])
    with cols[0]:
        go = st.button("Fetch captions", disabled=not yt_url)
    with cols[1]:
        dl = st.button("Download & index visuals", disabled=not yt_url)

    if go:
        try:
            vid = parse_video_id(yt_url)
        except ValueError as e:
            st.error(str(e)); st.stop()

        with st.spinner("Fetching captions..."):
            items = fetch_transcript_for_video(vid)  # [(ts, text)] or []
        if not items:
            st.warning("No captions found or access blocked. Use the Upload tab to transcribe with AssemblyAI.")
        else:
            with st.spinner("Chunking + embedding..."):
                chunks = chunk_transcript(items)
                texts = [c[1] for c in chunks]
                embs = embed_texts(texts)
                st.session_state.idx_chunks = chunks
                st.session_state.idx_embs = embs
                st.session_state.source = f"YouTube:{vid}"
            st.success(f"Indexed {len(chunks)} chunks from captions.")

    if dl:
        with st.spinner("Downloading MP4 (no cookies)..."):
            vpath = try_download_youtube_mp4(yt_url)
        if not vpath:
            st.error("Could not download video (network/corporate policy). Upload the file instead.")
        else:
            build_visual_index_from_video(vpath, every_sec=2.0)

with upload_tab:
    st.write("Upload an **audio/video** file (mp3, wav, m4a, mp4, mkv, mov). It will be transcribed by AssemblyAI.")
    up = st.file_uploader("Choose file", type=["mp3", "wav", "m4a", "mp4", "mkv", "mov"])
    if st.button("Transcribe with AssemblyAI", disabled=not up):
        if not os.getenv("ASSEMBLYAI_API_KEY", ""):
            st.error("ASSEMBLYAI_API_KEY missing in .env"); st.stop()
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", up.name)
        with open(path, "wb") as f: f.write(up.read())
        with st.spinner("Transcribing with AssemblyAI..."):
            items = transcribe_with_assemblyai(path)
        with st.spinner("Chunking + embedding..."):
            chunks = chunk_transcript(items)
            texts = [c[1] for c in chunks]
            embs = embed_texts(texts)
            st.session_state.idx_chunks = chunks
            st.session_state.idx_embs = embs
            st.session_state.source = f"Upload:{up.name}"
        st.success(f"Indexed {len(chunks)} chunks from uploaded file.")

        ext = (up.name.rsplit(".", 1)[-1] or "").lower()
        if ext in {"mp4", "mkv", "mov"}:
            build_visual_index_from_video(path, every_sec=2.0)

# Small status chips
c1, c2, c3 = st.columns(3)
c1.markdown(f'<span class="chunk-badge">Text chunks: {len(st.session_state.idx_chunks)}</span>', unsafe_allow_html=True)
c2.markdown(f'<span class="chunk-badge">Frames indexed: {len(st.session_state.vis_frames)}</span>', unsafe_allow_html=True)
c3.markdown(f'<span class="chunk-badge">Source: {st.session_state.source or "—"}</span>', unsafe_allow_html=True)

# ============================== Q&A section ========================================
st.divider()
st.subheader("Ask a question")
q = st.text_input("Your question")

if q and st.session_state.idx_chunks and st.session_state.idx_embs:
    with st.spinner("Searching + answering..."):
        # ---- Text retrieval
        qv = embed_one(q)
        top_idx = retrieve(qv, st.session_state.idx_embs, top_k=4)
        ctx = "\n\n".join(
            f"[Chunk {i+1} @ {st.session_state.idx_chunks[i][0]}] {st.session_state.idx_chunks[i][1]}"
            for i in top_idx
        )

        # ---- Visual retrieval: (1) by semantic similarity, (2) by exact chunk timestamps
        vis_hits: list[tuple[str,str]] = []
        # (1) semantic
        if st.session_state.vis_embs:
            try:
                qv_vis = embed_text_mm(q)
                vis_ids = retrieve(qv_vis, st.session_state.vis_embs, top_k=2)
                vis_hits.extend([st.session_state.vis_frames[i] for i in vis_ids])
            except Exception:
                pass
        # (2) timestamp-aligned frames for each retrieved chunk
        for i in top_idx:
            ts = st.session_state.idx_chunks[i][0]
            m = nearest_frame_for_ts(ts, st.session_state.vis_frames)
            if m and m not in vis_hits:
                vis_hits.append(m)

    # ---- Visual evidence preview (up to 4 images)
    if vis_hits:
        st.caption("Visual evidence:")
        show = vis_hits[:4]
        cols = st.columns(len(show))
        for col, (ts, imgp) in zip(cols, show):
            col.image(imgp, caption=f"Visual Frame @ {ts}", use_container_width=True)
    # ---- Multi-modal answer
    sys_prompt = (
        "Answer using ONLY the provided transcript chunks and (optional) visual frames. "
        "If the answer isn't present, say: 'I don’t know based on the transcript and video visuals.' "
        "Cite chunk numbers and timestamps for text, and cite visual frame timestamps when used."
    )

    parts = [f"System instruction: {sys_prompt}", f"Context:\n{ctx}", f"Question: {q}"]
    # inject images if we have any
    if vis_hits:
        image_parts = []
        for ts, imgp in vis_hits[:2]:
            with open(imgp, "rb") as f:
                data = f.read()
            mime, _ = mimetypes.guess_type(imgp)
            image_parts.append({"mime_type": mime or "image/jpeg", "data": data})
        parts = [f"System instruction: {sys_prompt}", f"Context:\n{ctx}", *image_parts, f"Question: {q}"]

    model = pick_chat_model()
    resp = model.generate_content([{"role": "user", "parts": parts}])
    st.write(resp.text.strip() if hasattr(resp, "text") and resp.text else str(resp))

    # Evidence footer
    text_evidence = ", ".join(f"Chunk {i+1} @ {st.session_state.idx_chunks[i][0]}" for i in top_idx)
    vis_evidence = ", ".join(f"Visual Frame @ {ts}" for ts, _ in vis_hits) if vis_hits else "None"
    st.markdown(f"**Evidence**  \nText: {text_evidence}  \nVisual: {vis_evidence}")

elif q:
    st.info("Build an index first (YouTube captions or Upload).")
