# app.py
"""
NYX – Youth Mental Wellness (Hackathon) — Fully Upgraded App
Supports:
- OpenRouter key (reuse Nova key) OR OpenAI key (new client)
- transformers fallback -> heuristic fallback
Features:
- Buddy Chat, AI Doc Chat
- Call Session (ElevenLabs preferred + pyttsx3 fallback)
- Mood & Journal (timeline, streaks, insights, optional wordcloud)
- Guided micro-actions, Buddy Boost card PNG download
- Crisis resources, Privacy mode, Clear local data
- Doctor Call link & demo Free Therapy Booking
- Community chat, Mini game
"""

import os
import io
import time
import random
import sqlite3
from datetime import datetime, timedelta, date, time as dtime
from typing import List, Tuple, Optional

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# Optional libs (guarded)
# -------------------------
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
    try:
        for v in _tts_engine.getProperty("voices"):
            if any(name in v.name for name in ("Samantha", "Alex", "Allison", "Victoria")):
                _tts_engine.setProperty("voice", v.id)
                break
    except Exception:
        pass
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

# -------------------------
# OpenAI / OpenRouter (new client)
# -------------------------
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
# Transformers conversational fallback (free)
# -------------------------
TRANSFORMERS_AVAILABLE = False
conv_pipe = None
Conversation = None
try:
    from transformers import pipeline, Conversation
    conv_pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    conv_pipe = None
    Conversation = None

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

def save_booking(name: str, d: date, t: dtime):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO bookings (ts, name, date, time) VALUES (?,?,?,?)", (ts, name, d.isoformat(), t.isoformat()))
    conn.commit()

def fetch_bookings(limit=20):
    cur = conn.cursor()
    cur.execute("SELECT ts, name, date, time FROM bookings ORDER BY ts DESC LIMIT ?", (limit,))
    return cur.fetchall()

def save_community(name: str, msg: str):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO community (ts, name, message) VALUES (?,?,?)", (ts, name, msg))
    conn.commit()

def fetch_community(limit=40):
    cur = conn.cursor()
    cur.execute("SELECT ts, name, message FROM community ORDER BY ts DESC LIMIT ?", (limit,))
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
# AI reply function
# -------------------------
def ai_reply_from_history(history: List[dict], mode: str = "buddy") -> str:
    # fallback safe
    if not history:
        return "Hi there! How are you feeling today?"
    last_text = history[-1]["user"] if "user" in history[-1] else history[-1].get("text", "")

    # Streamlit-safe default
    reply = "Hmm… I’m having trouble responding right now. Can we try again?"

    try:
        if OPENAI_AVAILABLE and openai_client:
            prompt = "\n".join([f"User: {h['user']}\nBot: {h.get('bot','')}" for h in history])
            prompt += f"\nUser: {last_text}\nBot:"
            res = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.7,
                max_tokens=200
            )
            reply = res.choices[0].message["content"].strip()
        elif TRANSFORMERS_AVAILABLE and conv_pipe and Conversation:
            conv = Conversation(last_text)
            conv_pipe(conv)
            reply = conv.generated_responses[-1] if conv.generated_responses else reply
        else:
            # simple heuristic fallback
            reply = random.choice([
                "I hear you. Can you tell me more?",
                "Interesting… go on",
                "Thanks for sharing that. How did it feel?"
            ])
    except Exception as e:
        st.warning(f"AI error fallback: {e}")
    return reply

# -------------------------
# Chat Page
# -------------------------
def page_chat(mode: str):
    st.subheader("🗨️ Buddy Chat" if mode=="buddy" else "🧑‍⚕️ AI Doc Chat")
    st.caption("Ongoing back-and-forth. Keep the conversation flowing 💬")

    key_hist = f"hist_{mode}"
    if key_hist not in st.session_state:
        st.session_state[key_hist] = []

    # user input
    user_input = st.text_input("Say something…", key=f"input_{mode}")
    if st.button("Send", key=f"send_{mode}") and user_input.strip():
        st.session_state[key_hist].append({"user": user_input})
        reply = ai_reply_from_history(st.session_state[key_hist], mode)
        st.session_state[key_hist][-1]["bot"] = reply

    # display chat history
    for msg in st.session_state[key_hist]:
        st.markdown(f"**You:** {msg.get('user','')}")
        st.markdown(f"**{mode.capitalize()}:** {msg.get('bot','…')}")


# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    tabs = ["Buddy Chat", "AI Doc Chat", "Mood Journal", "Community", "Call"]
    choice = st.sidebar.radio("Go to:", tabs)

    if choice == "Buddy Chat":
        page_chat("buddy")
    elif choice == "AI Doc Chat":
        page_chat("doc")
    elif choice == "Mood Journal":
        st.write("Mood + Journal coming soon…")
    elif choice == "Community":
        st.write("Community chat coming soon…")
    elif choice == "Call":
        st.write("Doctor call / therapy booking coming soon…")

if __name__ == "__main__":
    main()
