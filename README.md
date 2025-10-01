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
youtube-gemini-chat/
├── app.py # Streamlit UI + main logic
├── services/
│ ├── youtube.py # YouTube transcript API
│ ├── video.py # Frame extraction
│ └── assembly_ai.py # AssemblyAI integration
├── frames/ # Auto-saved visual frames
├── uploads/ # User-uploaded video files
├── requirements.txt # All Python dependencies
├── .gitignore # Prevents uploads/ & .env from being tracked
└── .env # Your API keys (keep private)

---

## 📦 Setup & Installation

### 🔹 Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-gemini-chat.git
cd youtube-gemini-chat

🔹 Install Python Requirements
pip install -r requirements.txt

🔹 Create a .env File

Inside the project root:

ASSEMBLYAI_API_KEY=your_assemblyai_key
GOOGLE_API_KEY=your_google_gemini_or_openai_key
💡 Usage
🔹 Start the App
streamlit run app.py

App will run on: http://localhost:8501

🔹 Upload a file or paste a YouTube link

The app will:

Transcribe the audio via AssemblyAI

Extract visual frames using OpenCV

Generate image embeddings using CLIP

Let you ask natural language questions about the video content

👩‍💻 Author

Made with ❤️ by Varsha Salimath
