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
````markdown
# 🎬 YouTube-Gemini Chat: Multi-Modal Video Understanding with Streamlit

An interactive Streamlit app that allows users to upload or link a YouTube video, then ask questions about it using both audio and visual context. This app integrates transcription, frame extraction, image embeddings, and natural language understanding to enable conversational search over video content.

---

## 🧠 How It Works

📺 **Input Video** → 🎧 **Audio Transcription (AssemblyAI)** + 🖼️ **Frame Extraction (OpenCV)**  
➡️ ✨ **Image Embeddings (CLIP)** + 🧠 **Gemini-based QA over transcript & visuals**  
➡️ 💬 **Answer your questions in natural language**

---

## 📁 Project Structure

```text
youtube-gemini-chat/
├── app.py              # Streamlit UI + main logic
├── services/           # Core logic modules
│   ├── youtube.py      # YouTube transcript API
│   ├── assembly_ai.py  # AssemblyAI transcription
│   ├── video.py        # Frame extraction & video utils
├── frames/             # Auto-saved visual frames
├── uploads/            # User-uploaded video/audio
├── .env                # API keys (DO NOT SHARE)
├── .gitignore          # Prevents uploads & secrets from being committed
├── requirements.txt    # All dependencies
````

---

## ⚙️ Setup & Installation

### 🔁 Clone the Repository

```bash
git clone https://github.com/Varsha-salimath/youtube-gemini-chat.git
cd youtube-gemini-chat
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔑 Create a `.env` File

Inside your project folder:

```env
ASSEMBLYAI_API_KEY=your_assemblyai_key
GOOGLE_API_KEY=your_google_gemini_or_openai_key
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🎥 Usage Flow

1. Upload a video/audio file or paste a YouTube link.
2. App will:

   * Transcribe audio via **AssemblyAI**
   * Extract visual frames using **OpenCV**
   * Generate embeddings using **CLIP**
   * Use **Gemini/OpenAI** to answer your questions
3. Ask questions like:

   * "What is this video about?"
   * "What’s happening around 2:30 mark visually?"
   * "Summarize the key discussion"

---


---

## 🌍 Deploy on Streamlit Cloud

1. Push your project to GitHub (excluding `uploads/` and `.env`)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → Click “New App”
3. Connect GitHub → Select repo → Click Deploy
4. Done! App will be accessible globally.

---

## 🧙‍♀️ Creator

> Made with 💡 by
## 
> **🚀 Varsha Salimath**
> *Because your videos deserve conversations too.*

---
