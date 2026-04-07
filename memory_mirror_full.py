# memory_mirror_full.py
import streamlit as st
from datetime import datetime
import sqlite3
import os
import tempfile
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

# Optional model imports (wrapped to fail gracefully)
USE_HF = False
try:
    from transformers import pipeline
    # don't instantiate heavy pipelines until needed (on-demand)
    USE_HF = True
except Exception:
    USE_HF = False

# -------------------------
# App Config & Paths
# -------------------------
st.set_page_config(page_title="AI Memory Mirror — Full App", layout="wide")
DB_PATH = "memory_mirror.db"
ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------
# Database Helpers
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_type TEXT,
        text_content TEXT,
        mood TEXT,
        summary TEXT,
        image_path TEXT,
        audio_path TEXT
    )''')
    conn.commit()
    return conn

conn = init_db()

def save_entry(entry):
    c = conn.cursor()
    c.execute('''
    INSERT INTO entries (timestamp, input_type, text_content, mood, summary, image_path, audio_path)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (entry['timestamp'], entry['input_type'], entry['text_content'], entry['mood'],
          entry['summary'], entry.get('image_path'), entry.get('audio_path')))
    conn.commit()
    return c.lastrowid

def fetch_entries(limit=500):
    c = conn.cursor()
    c.execute('SELECT id, timestamp, input_type, text_content, mood, summary, image_path, audio_path FROM entries ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    cols = ['id','timestamp','input_type','text_content','mood','summary','image_path','audio_path']
    return pd.DataFrame(rows, columns=cols)

# -------------------------
# AI Utilities
# -------------------------
# Summarizer (on-demand, to avoid heavy load)
summarizer = None
sentiment_pipeline = None
asr_pipeline = None

def get_summarizer():
    global summarizer
    if summarizer is None and USE_HF:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception:
            summarizer = None
    return summarizer

def get_sentiment_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None and USE_HF:
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
        except Exception:
            sentiment_pipeline = None
    return sentiment_pipeline

def summarize_text(text, prefer_quality=False):
    # prefer_quality=True will try HF summarizer if available
    if prefer_quality and USE_HF:
        summ = get_summarizer()
        if summ:
            try:
                out = summ(text, max_length=60, min_length=10, do_sample=False)
                return out[0]['summary_text']
            except Exception:
                pass
    # Lightweight fallback: rule-based short extract (first 2 sentences)
    sentences = text.strip().split('.')
    summary = '.'.join(sentences[:2]).strip()
    if len(summary)==0:
        return text[:150] + ("..." if len(text)>150 else "")
    return summary

def detect_text_mood(text, prefer_quality=False):
    # prefer HF sentiment if chosen and available
    if prefer_quality and USE_HF:
        pipe = get_sentiment_pipeline()
        if pipe:
            try:
                res = pipe(text[:512])[0]
                label = res['label'].lower()
                score = res.get('score', 0)
                if 'positive' in label:
                    return "😊 Positive"
                if 'negative' in label:
                    return "😔 Negative"
            except Exception:
                pass
    # Simple TextBlob fallback
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "😊 Positive"
    elif polarity < -0.2:
        return "😔 Negative"
    else:
        return "😐 Neutral"

def detect_image_mood_preset(pil_image):
    # Very simple: Haar cascade smile detector; not perfect but works offline
    try:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return "😐 No face detected"
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)
            if len(smiles) > 0:
                return "😊 Happy"
        return "😔 Sad/Neutral"
    except Exception as e:
        return "😐 Unknown (error)"

# Optional: audio transcription using SpeechRecognition (fallback)
import speech_recognition as sr

def transcribe_audio_file(file_bytes, prefer_quality=False):
    # Try to use whisper if installed and requested (not enforced)
    try:
        if prefer_quality and USE_HF:
            # attempt to use transformers' automatic-speech-recognition
            global asr_pipeline
            if asr_pipeline is None:
                try:
                    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
                except Exception:
                    asr_pipeline = None
            if asr_pipeline:
                # save tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(file_bytes)
                tmp.flush()
                tmp.close()
                res = asr_pipeline(tmp.name)
                os.unlink(tmp.name)
                return res.get('text','')
    except Exception:
        pass
    # Fallback: use SpeechRecognition with pocketsphinx or Google Web API (internet required)
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_name = tmp.name
        with sr.AudioFile(tmp_name) as source:
            audio = r.record(source)
        os.unlink(tmp_name)
        # try Google Web Speech API (requires internet, but no key for short use)
        try:
            return r.recognize_google(audio)
        except Exception:
            return ""
    except Exception:
        return ""

# -------------------------
# UI Helper: gratitude/suggestion
# -------------------------
def make_suggestion(mood):
    if "Positive" in mood or "Happy" in mood:
        return "Keep this energy — write 1 small achievement today."
    if "Negative" in mood or "Sad" in mood:
        return "Take a three-minute breathing break and list one small comfort."
    if "Neutral" in mood:
        return "Try a micro-joy activity tomorrow: 5-minute walk, music, or tea."
    if "No face detected" in mood:
        return "No face detected — try a closer selfie, or use text/audio."
    return "Consider writing 1 gratitude item."

# -------------------------
# PDF Export Helper
# -------------------------
def export_entry_to_pdf(entry):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10, "AI Memory Mirror — Exported Entry", ln=True, align='C')
    pdf.ln(5)
    pdf.cell(0,8, f"Timestamp: {entry['timestamp']}", ln=True)
    pdf.cell(0,8, f"Input Type: {entry['input_type']}", ln=True)
    pdf.cell(0,8, f"Mood: {entry['mood']}", ln=True)
    pdf.ln(3)
    pdf.multi_cell(0,8, f"Summary: {entry['summary']}")
    pdf.ln(3)
    pdf.multi_cell(0,8, f"Text Content:\n{entry['text_content']}")
    # include image if exists
    if entry.get('image_path') and os.path.exists(entry['image_path']):
        pdf.add_page()
        pdf.image(entry['image_path'], x=10, y=20, w=180)
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# -------------------------
# Streamlit App Layout
# -------------------------
st.title("🪞 AI Memory Mirror — Full Application")
st.markdown("Combine text, image, and audio journaling. Save entries, view timeline, and export PDFs.")

# Sidebar: settings and actions
st.sidebar.header("Settings & Quick Actions")
quality = st.sidebar.selectbox("Model quality preference", options=["Lightweight (fast)", "High-quality (slower)"])
prefer_quality = (quality == "High-quality (slower)")
if USE_HF:
    st.sidebar.write("HuggingFace available.")
else:
    st.sidebar.write("HuggingFace not available — using lightweight fallbacks.")

if st.sidebar.button("View Raw DB (debug)"):
    st.write(fetch_entries(200))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Journal", "Upload Image/Audio", "Memory Board & Timeline", "Manage / Export"])

# -------------------------
# Tab 1: Journal (Text)
# -------------------------
with tab1:
    st.header("Write your journal entry")
    text_input = st.text_area("Describe your day, thoughts, or a memory:", height=200)
    submit_text = st.button("Analyze & Save Text Entry")
    if submit_text:
        if not text_input.strip():
            st.warning("Please write something first.")
        else:
            ts = datetime.utcnow().isoformat()
            mood = detect_text_mood(text_input, prefer_quality=prefer_quality)
            summary = summarize_text(text_input, prefer_quality=prefer_quality)
            suggestion = make_suggestion(mood)

            # Save
            entry = {
                'timestamp': ts,
                'input_type': 'text',
                'text_content': text_input,
                'mood': mood,
                'summary': summary,
                'image_path': None,
                'audio_path': None
            }
            eid = save_entry(entry)
            st.success("Saved entry ✅")
            st.subheader("AI Insights")
            st.write("**Mood:**", mood)
            st.write("**Summary:**", summary)
            st.write("**Suggestion:**", suggestion)
            st.write(f"Entry ID: {eid}")

# -------------------------
# Tab 2: Upload Image/Audio
# -------------------------
with tab2:
    st.header("Upload image selfie or audio voice note")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image (Selfie) mood detection")
        uploaded_image = st.file_uploader("Upload a selfie (png/jpg)", type=['png','jpg','jpeg'], key="img1")
        if uploaded_image is not None:
            try:
                pil_img = Image.open(uploaded_image).convert("RGB")
                st.image(pil_img, caption="Uploaded image", use_column_width=True)
                img_mood = detect_image_mood_preset(pil_img)
                st.write("**Detected Mood:**", img_mood)
                st.write("**Suggestion:**", make_suggestion(img_mood))
                if st.button("Save image entry"):
                    # save image to disk
                    ts = datetime.utcnow().isoformat()
                    fname = f"img_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
                    path = os.path.join(ASSETS_DIR, fname)
                    pil_img.save(path)
                    entry = {
                        'timestamp': ts,
                        'input_type': 'image',
                        'text_content': '',
                        'mood': img_mood,
                        'summary': '',
                        'image_path': path,
                        'audio_path': None
                    }
                    eid = save_entry(entry)
                    st.success(f"Saved image entry (ID {eid})")
            except Exception as e:
                st.error("Error processing image: " + str(e))


    with col2:
        st.subheader("Audio (voice) — upload .wav or .mp3")
        uploaded_audio = st.file_uploader("Upload audio file (.wav .mp3)", type=['wav','mp3','m4a'], key="aud1")
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            st.audio(audio_bytes)
            with st.spinner("Transcribing (may take time)..."):
                transcription = transcribe_audio_file(audio_bytes, prefer_quality=prefer_quality)
            if transcription:
                st.write("**Transcription:**", transcription)
                mood = detect_text_mood(transcription, prefer_quality=prefer_quality)
                summary = summarize_text(transcription, prefer_quality=prefer_quality)
                st.write("**Detected Mood:**", mood)
                st.write("**Summary:**", summary)
                if st.button("Save audio entry"):
                    ts = datetime.utcnow().isoformat()
                    fname = f"aud_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.wav"
                    path = os.path.join(ASSETS_DIR, fname)
                    with open(path, "wb") as f:
                        f.write(audio_bytes)
                    entry = {
                        'timestamp': ts,
                        'input_type': 'audio',
                        'text_content': transcription,
                        'mood': mood,
                        'summary': summary,
                        'image_path': None,
                        'audio_path': path
                    }
                    eid = save_entry(entry)
                    st.success(f"Saved audio entry (ID {eid})")
            else:
                st.warning("Transcription failed or empty. Try another file or enable high-quality models if available.")

# -------------------------
# Tab 3: Memory Board & Timeline
# -------------------------
with tab3:
    st.header("Memory Board & Mood Timeline")
    df = fetch_entries(500)
    if df.empty:
        st.info("No entries yet — add text, image or audio.")
    else:
        # show memory board (cards)
        st.subheader("Recent Memories")
        for idx, row in df.head(12).iterrows():
            with st.expander(f"Entry {row['id']} — {row['timestamp']} — {row['mood']}"):
                st.write("Type:", row['input_type'])
                if row['text_content']:
                    st.write("Text:", row['text_content'])
                if row['summary']:
                    st.write("Summary:", row['summary'])
                if row['image_path']:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], width=300)
                if row['audio_path']:
                    if os.path.exists(row['audio_path']):
                        st.audio(row['audio_path'])

        # timeline chart
        st.subheader("Mood Timeline (last 100 entries)")
        plot_df = df.copy()
        if not plot_df.empty:
            # map moods to numeric for plotting
            def mood_to_num(m):
                if isinstance(m, str):
                    if "Positive" in m or "Happy" in m:
                        return 2
                    if "Neutral" in m:
                        return 1
                    if "Negative" in m or "Sad" in m:
                        return 0
                return 1
            plot_df['ts'] = pd.to_datetime(plot_df['timestamp'], errors='coerce')
            plot_df = plot_df.sort_values('ts')
            plot_df['mood_num'] = plot_df['mood'].apply(mood_to_num)
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(plot_df['ts'], plot_df['mood_num'], marker='o', linewidth=1)
            ax.set_yticks([0,1,2]); ax.set_yticklabels(['Sad/Neg','Neutral','Happy/Pos'])
            ax.set_title("Mood over time")
            ax.set_xlabel("Timestamp"); ax.set_ylabel("Mood")
            st.pyplot(fig)

# -------------------------
# Tab 4: Manage / Export
# -------------------------
with tab4:
    st.header("Manage & Export Entries")
    df_all = fetch_entries(1000)
    st.dataframe(df_all[['id','timestamp','input_type','mood']])
    selected_id = st.number_input("Enter Entry ID to view/export", min_value=1, step=1, value=0)
    if selected_id:
        row = df_all[df_all['id']==selected_id]
        if row.empty:
            st.warning("Entry not found.")
        else:
            entry = row.iloc[0].to_dict()
            st.write(entry)
            if st.button("Export entry to PDF"):
                buf = export_entry_to_pdf(entry)
                st.download_button("Download PDF", data=buf, file_name=f"entry_{selected_id}.pdf", mime="application/pdf")
            if st.button("Delete entry (permanent)"):
                c = conn.cursor()
                c.execute("DELETE FROM entries WHERE id=?", (selected_id,))
                conn.commit()
                st.success("Deleted (refresh to see changes).")

# Footer
st.markdown("---")
st.caption("AI Memory Mirror — Prototype. Models and detection are for educational/demo use. For production, consider stronger models, privacy protections, and informed user consent for biometric processing.")
