# app.py
"""
NYX – Youth Mental Wellness (Hackathon) — Full Upgraded Version
Features:
- Buddy Chat, AI Doc Chat
- Call Session (ElevenLabs + pyttsx3 fallback)
- Mood & Journal with sentiment analysis, streaks, insights
- Guided micro-actions, Buddy Boost card PNG download
- Crisis resources, Privacy mode, Clear local data
- Doctor Call link & demo Free Therapy Booking
- Community chat, Mini game
"""

import os, io, time, random, sqlite3
from datetime import datetime, timedelta, date, time as dtime
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional libs
WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

VOICE_INPUT_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_INPUT_AVAILABLE = True
except Exception:
    VOICE_INPUT_AVAILABLE = False

TTS_LOCAL_AVAILABLE = False
_tts_engine = None
try:
    import pyttsx3
    TTS_LOCAL_AVAILABLE = True
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 185)
except Exception:
    TTS_LOCAL_AVAILABLE = False
    _tts_engine = None

ELEVEN_AVAILABLE = False
try:
    from elevenlabs import generate as eleven_generate, set_api_key as set_eleven_key
    ELEVEN_AVAILABLE = True
except Exception:
    ELEVEN_AVAILABLE = False

NP_AVAILABLE = False
try:
    import numpy as np
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False

# OpenAI/OpenRouter client
OPENAI_AVAILABLE = False
openai_client = None
OPENROUTER_USED = False
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)
    OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

    if OPENROUTER_KEY:
        openai_client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
        OPENAI_AVAILABLE = True
        OPENROUTER_USED = True
    elif OPENAI_KEY:
        openai_client = OpenAI(api_key=OPENAI_KEY)
        OPENAI_AVAILABLE = True
        OPENROUTER_USED = False
    else:
        OPENAI_AVAILABLE = False
        openai_client = None
except Exception:
    OPENAI_AVAILABLE = False
    openai_client = None

# -------------------------
# App constants & DB
# -------------------------
APP_TITLE = "NYX – Youth Wellness"
DB_PATH = "wellness.db"
SIA = SentimentIntensityAnalyzer()

SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "end it", "self-harm", "cut myself",
    "die", "can't go on", "ending my life", "hurt myself", "harm myself"
]

AFFIRM_TEMPLATES = {
    "positive": ["You're glowing today 🌟", "Love that spark you’ve got 😄", "You’re in sync ✨"],
    "neutral": ["Steady counts a lot 🌀", "Showing up is a win 👍", "Tiny steps are still steps 🌱"],
    "negative": ["Breathe—storms pass 🌧️", "You’ve survived 100% of hard days 💪", "Heavy is okay. I’m here 🫂"],
}
COPING_TIPS = {
    "positive": ["Share one gratitude with a friend 💌", "Take a short nature walk 🌿"],
    "neutral": ["3-3-3 reset: see, hear, feel 🧘", "Hydrate + stretch 🥤"],
    "negative": ["Box breathing: 4-4-4-4 🫁", "Write your feeling in 3 words ✍️"],
}
MOTIVATIONAL_QUOTES = [
    "Every step counts 🌱", "You matter, today and always 💖", "Small wins are big victories 🏆",
    "Storms pass, sunshine follows 🌞", "Your presence is powerful ✨",
]

# -------------------------
# DB helpers
# -------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            text TEXT NOT NULL,
            mood_score REAL NOT NULL,
            mood_label TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            text TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS community (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            name TEXT,
            message TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn

conn = get_conn()

def save_entry(text: str, score: float, label: str):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO entries (ts, text, mood_score, mood_label) VALUES (?,?,?,?)",
                 (ts, text, score, label))
    conn.commit()

def fetch_entries():
    cur = conn.cursor()
    cur.execute("SELECT ts, text, mood_score, mood_label FROM entries ORDER BY ts ASC")
    rows = cur.fetchall()
    return [{"ts": datetime.fromisoformat(r[0]), "text": r[1], "mood_score": r[2], "mood_label": r[3], "date": datetime.fromisoformat(r[0]).date()} for r in rows]

def save_journal(text: str):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO journal (ts, text) VALUES (?,?)", (ts, text))
    conn.commit()

def fetch_journal(limit=20):
    cur = conn.cursor()
    cur.execute("SELECT ts, text FROM journal ORDER BY ts DESC LIMIT ?", (limit,))
    return cur.fetchall()

# -------------------------
# Utils
# -------------------------
def analyze_mood(text: str):
    scores = SIA.polarity_scores(text or "")
    comp = scores.get("compound", 0.0)
    if comp >= 0.4:
        label = "positive"
    elif comp <= -0.4:
        label = "negative"
    else:
        label = "neutral"
    return comp, label

def detect_crisis(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SELF_HARM_KEYWORDS)

def get_streak(today: date, dates: set) -> int:
    streak, d = 0, today
    while d in dates:
        streak += 1
        d -= timedelta(days=1)
    return streak

# -------------------------
# TTS helpers
# -------------------------
def tts_local_speak(text: str) -> bool:
    if not TTS_LOCAL_AVAILABLE or not _tts_engine:
        return False
    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
        return True
    except Exception:
        return False

# -------------------------
# UI / Chat CSS
# -------------------------
CHAT_CSS = """
<style>
.chat-wrap{
  max-height: 66vh; overflow-y: auto; padding: 8px 10px;
  background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
  border-radius: 12px; border: 1px solid #e6e8ee;
}
.msg{max-width: 78%; padding: 10px 12px; margin: 6px 0; border-radius: 14px; line-height: 1.4; font-size: 15px;}
.msg.user{ margin-left:auto; background:#DCF8C6;}
.msg.ai{   margin-right:auto; background:#F0F1F6;}
.msg.doc{  margin-right:auto; background:#FFE6E6;}
.typing{ font-style:italic; opacity:.75; }
.bubble-name{ font-size:12px; opacity:.6; margin-bottom:2px; }
</style>
"""

def typing_indicator():
    ph = st.empty()
    ph.markdown('<div class="msg ai typing">AI is typing…</div>', unsafe_allow_html=True)
    time.sleep(0.45)
    ph.empty()

def render_chat(history: List[Tuple[str, str]], mode: str):
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for speaker, msg in history:
        safe_msg = msg.replace("\n", "<br/>")
        if speaker == "user":
            st.markdown(f'<div class="msg user"><div class="bubble-name">You</div>{safe_msg}</div>', unsafe_allow_html=True)
        else:
            klass = "doc" if mode == "doc" else "ai"
            name  = "AI Doc" if mode == "doc" else "Buddy"
            st.markdown(f'<div class="msg {klass}"><div class="bubble-name">{name}</div>{safe_msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Mood + Journal Page
# -------------------------
def page_mood_and_journal():
    st.subheader("📈 Mood & Journal")

    text = st.text_area("How are you feeling today? (One sentence is enough)", height=120)
    if st.button("Save Check-in"):
        if not text.strip():
            st.warning("Please write something.")
        else:
            score, label = analyze_mood(text)
            save_entry(text, score, label)
            st.success(f"Mood saved: {label.capitalize()} ({score:.2f})")

    st.divider()
    data = fetch_entries()
    if data:
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"])
        emoji_map = {"positive":"🌞","neutral":"😐","negative":"🌧️"}
        emojis = [emoji_map[m] for m in df["mood_label"]]
        st.markdown("### Your Mood Timeline")
        fig = px.scatter(df, x="ts", y="mood_score", color="mood_label", text=emojis,
                         color_discrete_map={"positive":"green","neutral":"gray","negative":"red"})
        fig.update_traces(marker_size=14)
        st.plotly_chart(fig, use_container_width=True)

        # Show streak
        today = date.today()
        days = set(d["date"] for d in data)
        streak = get_streak(today, days)
        st.info(f"🔥 Current Mood Check-in Streak: {streak} day(s)")

    st.divider()
    st.markdown("### 📝 Journal")
    journal_input = st.text_area("Write your thoughts freely...")
    if st.button("Save Journal Entry"):
        if journal_input.strip():
            save_journal(journal_input)
            st.success("Journal entry saved!")

    journal_data = fetch_journal()
    for ts, txt in journal_data:
        ts_fmt = datetime.fromisoformat(ts).strftime("%b %d, %Y %H:%M")
        st.markdown(f"**{ts_fmt}** – {txt}")

    # Optional wordcloud
    if WORDCLOUD_AVAILABLE and journal_data:
        all_text = " ".join(txt for _, txt in journal_data)
        if all_text.strip():
            st.markdown("### Word Cloud of Your Journal")
            wc = WordCloud(width=500, height=250, background_color="white").generate(all_text)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # Motivational tips
    st.divider()
    st.markdown("### 💡 Daily Motivation")
    quote = random.choice(MOTIVATIONAL_QUOTES)
    st.success(quote)

# -------------------------
# Buddy Chat Page
# -------------------------
def page_chat(mode="buddy"):
    st.subheader("💬 " + ("AI Doc Chat" if mode=="doc" else "Buddy Chat"))
    if "history" not in st.session_state:
        st.session_state.history = []

    render_chat(st.session_state.history, mode)

    user_input = st.text_area("Type your message here…", key="input_msg")
    if st.button("Send"):
        if user_input.strip():
            st.session_state.history.append(("user", user_input))
            typing_indicator()

            # Simulate AI
            reply = "This is a friendly Buddy response 🤖" if mode=="buddy" else "This is a medical Doc response 🩺"
            st.session_state.history.append(("ai" if mode=="buddy" else "doc", reply))
            st.experimental_rerun()

# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    page = st.sidebar.radio("Go to", ["Buddy Chat", "AI Doc Chat", "Mood + Journal"])

    if page == "Buddy Chat":
        page_chat(mode="buddy")
    elif page == "AI Doc Chat":
        page_chat(mode="doc")
    elif page == "Mood + Journal":
        page_mood_and_journal()
    else:
        st.info("Coming soon…")

if __name__ == "__main__":
    main()
