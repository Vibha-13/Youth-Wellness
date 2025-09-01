# app_streamlit_ready.py
"""
NYX – Youth Mental Wellness (Hackathon) — Streamlit-safe version
TTS / Voice input dependencies optional to avoid installation errors
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

# Mic/Voice input is now optional with safe fallback
VOICE_INPUT_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_INPUT_AVAILABLE = True
except Exception:
    VOICE_INPUT_AVAILABLE = False

# Local TTS fallback
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
# OpenAI / OpenRouter
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
# Transformers conversational fallback (optional)
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
# Constants & DB
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

# ... all your DB save/fetch functions remain the same ...
# analyze_mood, detect_crisis, streak, ai_reply_from_history (unchanged) ...

# -------------------------
# TTS helpers with safe fallbacks
# -------------------------
def tts_elevenlabs_bytes(text: str, voice: str = "Rachel") -> Optional[bytes]:
    if not ELEVEN_AVAILABLE:
        return None
    try:
        return eleven_generate(text=text, voice=voice, model="eleven_multilingual_v1")
    except Exception:
        return None

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
# UI / Chat & all pages remain same
# -------------------------
# You can copy all functions for rendering chat, guided actions, mood dashboard,
# buddy card, mini-game, journaling, doctor booking, crisis resources, etc.
# Only difference is VOICE_INPUT_AVAILABLE now safely False if not installed.

# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🫶", layout="wide")
    st.markdown(f"<h1 style='text-align:center'>{APP_TITLE} 🫶</h1>", unsafe_allow_html=True)
    st.sidebar.title("NYX Controls")
    st.sidebar.write("OpenAI configured:", "✅" if OPENAI_AVAILABLE else "⚠️ Not configured")
    st.sidebar.write("Transformers conv:", "✅" if TRANSFORMERS_AVAILABLE else "⚠️ Not installed")
    st.sidebar.write("ElevenLabs (TTS):", "✅" if ELEVEN_AVAILABLE else "⚠️ Missing/Not installed")
    st.sidebar.write("Local TTS (pyttsx3):", "✅" if TTS_LOCAL_AVAILABLE else "⚠️ Not available")
    st.sidebar.write("Mic Input:", "✅" if VOICE_INPUT_AVAILABLE else "⚠️ Optional — install PyAudio for full voice")
    
    # Sidebar controls & private mode
    if "private_mode" not in st.session_state:
        st.session_state["private_mode"] = False
    st.session_state["private_mode"] = st.sidebar.checkbox("Private Mode (don't save check-ins)", value=st.session_state["private_mode"])
    if st.sidebar.button("Clear All Local Data"):
        conn.execute("DELETE FROM entries"); conn.execute("DELETE FROM journal")
        conn.execute("DELETE FROM community"); conn.execute("DELETE FROM bookings"); conn.commit()
        st.sidebar.success("All local data cleared")

    # Crisis resources
    with st.sidebar.expander("🚨 Crisis Resources", expanded=True):
        st.write("If you're in immediate danger, call your local emergency number.")
        st.write("- India (example): 112")
        st.write("This app is not a substitute for professional care.")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        [
            "🗨️ Buddy Chat",
            "🧑‍⚕️ AI Doc Chat",
            "📞 Call Session (AI Doc)",
            "📈 Mood & Journal",
            "🎮 Mini Game",
            "📓 Journal & Community",
            "👩‍⚕️ Doctor Call & Booking",
        ],
    )

    # Dispatch to pages (reuse your existing functions)
    if page == "🗨️ Buddy Chat":
        page_chat(mode="buddy")
    elif page == "🧑‍⚕️ AI Doc Chat":
        page_chat(mode="doc")
    elif page == "📞 Call Session (AI Doc)":
        page_call_session()
    elif page == "📈 Mood & Journal":
        page_mood_and_journal()
    elif page == "🎮 Mini Game":
        page_game()
    elif page == "📓 Journal & Community":
        page_journal_and_community()
    elif page == "👩‍⚕️ Doctor Call & Booking":
        page_doctor_call_and_booking()
    else:
        st.info("Select a page from the sidebar.")

if __name__ == "__main__":
    main()
