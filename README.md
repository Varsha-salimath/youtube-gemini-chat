# 🎥 YouTube-Gemini Chat

A powerful **AI-powered multimodal Streamlit application** that allows users to interact with YouTube videos or uploaded MP4 files using natural language. It combines **AssemblyAI** for audio transcription, **CLIP** for frame embedding, and **Google Gemini/OpenAI** for answering your questions based on both **text and visuals** from the video.

---

## 🚀 Features

- 🎞️ Upload or analyze YouTube videos (`.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a`)
- 🧠 Auto-transcription using AssemblyAI
- 📸 Visual frame extraction using OpenCV
- 🔍 Embeds visual frames using CLIP (via sentence-transformers)
- 💬 Ask questions about the video; receive smart, visual+textual responses
- 🌐 Hosted on **Streamlit Cloud**
- 🛠️ Backend-agnostic — works with Google Gemini or OpenAI APIs

---

## 🧩 Tech Stack

| Component      | Technology               |
|----------------|---------------------------|
| UI             | Streamlit                 |
| Transcription  | AssemblyAI                |
| Visual Frames  | OpenCV                    |
| Image Embedding| CLIP via SentenceTransformer |
| LLM Backend    | Google Gemini / OpenAI    |
| File Upload    | Streamlit file_uploader   |
| Deployment     | Streamlit Cloud           |

---

## 🧠 Architecture Diagram

<img width="1536" height="1024" alt="ChatGPT Image Oct 2, 2025, 02_50_49 AM" src="https://github.com/user-attachments/assets/2fe86d2c-bb95-4e71-ab77-088f258efae5" />

---

## 📁 Project Structure

