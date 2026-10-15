# 🎙️ VoiceClone AI

VoiceClone AI is an AI-powered voice cloning application that generates speech using a user's voice sample.

The project uses **Python, FastAPI, Streamlit, and Qwen3-TTS** to process a voice reference and generate AI speech.

## 🚀 Features

- 🎙️ Upload your own voice sample
- 🔊 Generate speech using the reference voice
- 🎧 Listen to generated audio
- ⬇️ Download generated voice
- 🔍 Analyze voice quality
- 📊 Audio duration and sample-rate analysis
- 🎚️ SNR (Signal-to-Noise Ratio) analysis
- 📝 Automatic reference-audio transcription
- 🌐 FastAPI backend
- 🖥️ Streamlit frontend

## 🛠️ Tech Stack

- Python
- FastAPI
- Streamlit
- PyTorch
- Qwen3-TTS
- Transformers
- Librosa
- NumPy
- SciPy
- SoundFile
- Whisper

## 📁 Project Structure

```text
voice-cloning-ai/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── audio.py
│   └── utils.py
│
├── data/
│   └── voice_samples/
│
├── outputs/
│
├── frontend/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
