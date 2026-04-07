🪞 AI Memory Mirror — Emotional Journal App
📌 Overview

AI Memory Mirror is a full-stack, AI-powered journaling application that helps users track, analyze, and understand their emotions through daily entries.

The app allows users to log their thoughts using text, images, or voice, and applies Artificial Intelligence (NLP-based sentiment analysis) to detect emotional states and provide meaningful insights.

It is built using Streamlit, making it lightweight, interactive, and easy to deploy.

🚀 Features
🔐 User Authentication
Secure login & signup system
Separate journal data for each user
🧠 AI Emotion Detection
Sentiment analysis using NLP
Detects Positive, Negative, or Neutral mood
✍️ Multi-Input Journal
Text journaling
Image upload (selfie mood detection)
Voice input (record/upload audio)
📊 Interactive Dashboard
Mood timeline visualization
Memory board showing past entries
📄 Export Functionality
Download journal entries as PDF
🎨 Modern UI/UX
Colorful, responsive interface
Smooth user interaction using Streamlit
🏗️ Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	Python
Database	SQLite
AI/NLP	TextBlob
Image Processing	OpenCV
Audio Processing	SpeechRecognition
PDF Export	FPDF
📂 Project Structure
AI-Memory-Mirror/
│
├── memory_mirror_final.py   # Main application
├── memory_mirror.db        # SQLite database (auto-created)
├── assets/                 # Saved images & audio
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/sathyasree07/AI-mirror.git
cd ai-memory-mirror
2️⃣ Create Virtual Environment
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run memory_mirror_final.py
📸 Usage
Create an account (Signup)
Login to your dashboard
Add journal entries via:
Text
Image
Voice
View emotional insights
Track mood over time
Export entries as PDF
🧠 How AI Works
Text Analysis:
Uses NLP sentiment analysis to classify emotions
Image Analysis:
Detects faces and expressions using computer vision
Audio Processing:
Converts speech → text → sentiment analysis
🔮 Future Enhancements
Advanced deep learning emotion detection
Personalized mental health suggestions
Cloud database integration
Mobile app version
Chatbot companion
