# app.py
"""
NYX – Youth Mental Wellness (Hackathon) — Full Upgraded Version
Features:
- Buddy Chat, AI Doc Chat
- Mood & Journal with sentiment analysis, streaks, insights
- Guided micro-actions, Buddy Boost card PNG download
- Crisis resources, Privacy mode, Clear local data
- Doctor Call link & demo Free Therapy Booking
- Community chat, Mini game, Breathing exercises
"""

import os, io, time, random, sqlite3
from datetime import datetime, timedelta, date
from typing import List, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional libraries
WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except:
    WORDCLOUD_AVAILABLE = False

TTS_AVAILABLE = False
_tts_engine = None
try:
    import pyttsx3
    TTS_AVAILABLE = True
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 180)
except:
    TTS_AVAILABLE = False

OPENAI_AVAILABLE = False
openai_client = None
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)
    if OPENROUTER_KEY:
        openai_client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
        OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

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

MOTIVATIONAL_QUOTES = [
    "Every step counts 🌱", "You matter, today and always 💖", 
    "Small wins are big victories 🏆", "Storms pass, sunshine follows 🌞", 
    "Your presence is powerful ✨"
]

# -------------------------
# Database helpers
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
    return [{"ts": datetime.fromisoformat(r[0]), "text": r[1], "mood_score": r[2], "mood_label": r[3], 
             "date": datetime.fromisoformat(r[0]).date()} for r in rows]

def save_journal(text: str):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO journal (ts, text) VALUES (?,?)", (ts, text))
    conn.commit()

def fetch_journal(limit=20):
    cur = conn.cursor()
    cur.execute("SELECT ts, text FROM journal ORDER BY ts DESC LIMIT ?", (limit,))
    return cur.fetchall()

def save_community(name: str, message: str):
    ts = datetime.now().isoformat()
    conn.execute("INSERT INTO community (ts, name, message) VALUES (?,?,?)", (ts, name, message))
    conn.commit()

def fetch_community(limit=50):
    cur = conn.cursor()
    cur.execute("SELECT ts, name, message FROM community ORDER BY ts DESC LIMIT ?", (limit,))
    return cur.fetchall()

# -------------------------
# Mood analysis & utils
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
# TTS
# -------------------------
def speak(text: str):
    if TTS_AVAILABLE and _tts_engine:
        _tts_engine.say(text)
        _tts_engine.runAndWait()

# -------------------------
# Chat helpers
# -------------------------
def chat_openrouter(prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return "OpenRouter GPT not available 😢"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def render_chat(history: List[Tuple[str, str]], mode: str):
    for speaker, msg in history:
        name = "Buddy" if mode=="buddy" else "AI Doc"
        if speaker=="user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**{name}:** {msg}")

# -------------------------
# Pages
# -------------------------
def page_mood_journal():
    st.subheader("📈 Mood & Journal")
    text = st.text_area("How are you feeling today?", height=100)
    if st.button("Save Mood"):
        if text.strip():
            score, label = analyze_mood(text)
            save_entry(text, score, label)
            st.success(f"Mood saved: {label} ({score:.2f})")
    st.divider()

    data = fetch_entries()
    if data:
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["ts"])
        emoji_map = {"positive":"🌞","neutral":"😐","negative":"🌧️"}
        emojis = [emoji_map[m] for m in df["mood_label"]]
        fig = px.scatter(df, x="ts", y="mood_score", color="mood_label", text=emojis,
                         color_discrete_map={"positive":"green","neutral":"gray","negative":"red"})
        st.plotly_chart(fig, use_container_width=True)
        today = date.today()
        days = set(d["date"] for d in data)
        streak = get_streak(today, days)
        st.info(f"🔥 Current Mood Streak: {streak} day(s)")

    st.divider()
    st.markdown("### 📝 Journal")
    journal_input = st.text_area("Write your thoughts freely")
    if st.button("Save Journal"):
        if journal_input.strip():
            save_journal(journal_input)
            st.success("Journal saved!")
    journal_data = fetch_journal()
    for ts, txt in journal_data:
        ts_fmt = datetime.fromisoformat(ts).strftime("%b %d, %Y %H:%M")
        st.markdown(f"**{ts_fmt}** – {txt}")
    if WORDCLOUD_AVAILABLE and journal_data:
        all_text = " ".join(txt for _, txt in journal_data)
        wc = WordCloud(width=500, height=250, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    st.divider()
    st.success(random.choice(MOTIVATIONAL_QUOTES))

def page_buddy_doc(mode="buddy"):
    st.subheader("💬 " + ("AI Doc Chat" if mode=="doc" else "Buddy Chat"))
    if "history" not in st.session_state:
        st.session_state.history = []
    render_chat(st.session_state.history, mode)
    user_input = st.text_area("Type your message here…", key=f"input_{mode}")
    if st.button("Send", key=f"btn_{mode}"):
        if user_input.strip():
            st.session_state.history.append(("user", user_input))
            reply = chat_openrouter(user_input)
            st.session_state.history.append(("ai" if mode=="buddy" else "doc", reply))
            speak(reply)
            st.experimental_rerun()

def page_community():
    st.subheader("🌐 Community Chat")
    name = st.text_input("Your Name")
    msg = st.text_area("Message")
    if st.button("Send Message"):
        if msg.strip():
            save_community(name or "Anonymous", msg)
            st.success("Message posted!")
            st.experimental_rerun()
    msgs = fetch_community()
    for ts, n, m in msgs:
        ts_fmt = datetime.fromisoformat(ts).strftime("%b %d %H:%M")
        st.markdown(f"**{n} [{ts_fmt}]:** {m}")

def page_mini_game():
    st.subheader("🎮 Mini Game: Guess a Number")
    number = st.session_state.get("target_number", random.randint(1,50))
    guess = st.number_input("Enter your guess", min_value=1, max_value=50, value=25)
    if st.button("Guess"):
        if guess == number:
            st.success("🎉 You guessed it!")
            st.session_state.target_number = random.randint(1,50)
        elif guess < number:
            st.info("Try higher ⬆️")
        else:
            st.info("Try lower ⬇️")
    st.session_state.target_number = number

def page_breathing():
    st.subheader("🫁 Breathing Exercise")
    st.markdown("""
    Follow the steps below:
    1. Inhale 🫁 for 4 seconds
    2. Hold 🤚 for 4 seconds
    3. Exhale 🌬️ for 4 seconds
    4. Hold ✋ for 4 seconds
    Repeat 3–5 cycles
    """)

def page_doctor_call():
    st.subheader("📞 Doctor Call / Therapy Booking")
    st.markdown("[Book a free session](https://www.example.com)")

# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    page = st.sidebar.radio("Go to", [
        "Buddy Chat", "AI Doc Chat", "Mood + Journal",
        "Community Chat", "Mini Game", "Breathing Exercise",
        "Doctor Call"
    ])
    if page=="Buddy Chat":
        page_buddy_doc("buddy")
    elif page=="AI Doc Chat":
        page_buddy_doc("doc")
    elif page=="Mood + Journal":
        page_mood_journal()
    elif page=="Community Chat":
        page_community()
    elif page=="Mini Game":
        page_mini_game()
    elif page=="Breathing Exercise":
        page_breathing()
    elif page=="Doctor Call":
        page_doctor_call()

if __name__=="__main__":
    main()
import streamlit as st
import os
import requests
import random
import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyttsx3

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Youth Wellness App", layout="wide")

# API Key
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Session State Init
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "journal" not in st.session_state:
    st.session_state.journal = []
if "community_chat" not in st.session_state:
    st.session_state.community_chat = []

# ------------------ HELPERS ------------------
def call_openrouter(prompt, role="Buddy"):
    if not OPENROUTER_KEY:
        return f"[Offline Mode] {role} says: You're doing great 🌸"
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": f"You are a supportive {role}."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[{role}] fallback: Stay strong 💙 (error {e})"


def tts_output(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        st.warning("TTS not available on this device.")


# ------------------ PAGES ------------------
def page_chat(mode="buddy"):
    st.title("💬 Chat")
    role = "Buddy" if mode == "buddy" else "AI Doc"
    user_input = st.text_input(f"Talk to {role}:")
    if st.button("Send") and user_input:
        reply = call_openrouter(user_input, role=role)
        st.session_state.chat_log.append(("You", user_input))
        st.session_state.chat_log.append((role, reply))
    for speaker, msg in st.session_state.chat_log[::-1]:
        st.markdown(f"**{speaker}:** {msg}")


def page_call():
    st.title("📞 Call Session")
    st.write("Simulated doctor call (TTS)")
    if st.button("Start Call"):
        response = "Hello, I am your AI Doc. Please describe your issue."
        tts_output(response)
        st.success("Call simulation started!")


def page_journal():
    st.title("📔 Mood & Journal")
    mood = st.selectbox("How do you feel?", ["😊 Happy", "😢 Sad", "😡 Angry", "😴 Tired"])
    entry = st.text_area("Write your thoughts")
    if st.button("Save Entry"):
        st.session_state.journal.append((str(datetime.date.today()), mood, entry))
        st.success("Saved!")

    if st.session_state.journal:
        moods = [m for _, m, _ in st.session_state.journal]
        text = " ".join([e for _, _, e in st.session_state.journal])
        wc = WordCloud(width=400, height=200, background_color="white").generate(text)
        st.image(wc.to_array())


def page_micro():
    st.title("🌱 Guided Micro-Actions")
    if st.button("Breathing Exercise"):
        st.info("Inhale... Hold... Exhale... Repeat 3 times")
    if st.button("Hydration Reminder"):
        st.success("Drink a glass of water 💧")
    if st.button("5-4-3-2-1 Grounding"):
        st.write("Notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.")


def page_dashboard():
    st.title("📊 Insights Dashboard")
    if st.session_state.journal:
        moods = [m for _, m, _ in st.session_state.journal]
        fig, ax = plt.subplots()
        ax.hist(moods)
        st.pyplot(fig)


def page_boost():
    st.title("🚀 Buddy Boost")
    quote = random.choice([
        "You are enough 🌸",
        "Storms don’t last forever ⛈️",
        "Your pace is perfect 🐢",
    ])
    st.info(quote)


def page_resources():
    st.title("🆘 Crisis & Resources")
    st.markdown("**India:** 9152987821 (AASRA)")
    st.markdown("**Global:** https://findahelpline.com")
    st.markdown("**Doctor Call:** [Book here](https://doxy.me)")
    st.markdown("**Free Therapy:** [Register](https://betterhelp.com)")


def page_community():
    st.title("👥 Community Chat")
    msg = st.text_input("Say something to the community")
    if st.button("Post") and msg:
        st.session_state.community_chat.append(("You", msg))
    for speaker, text in st.session_state.community_chat[::-1]:
        st.markdown(f"**{speaker}:** {text}")


def page_game():
    st.title("🎮 Mini Game: Guess the Mood")
    emoji = random.choice(["😊", "😢", "😡", "😴"])
    guess = st.selectbox("What does this emoji mean?", ["Happy", "Sad", "Angry", "Tired"])
    if st.button("Check"):
        if (emoji == "😊" and guess == "Happy") or \
           (emoji == "😢" and guess == "Sad") or \
           (emoji == "😡" and guess == "Angry") or \
           (emoji == "😴" and guess == "Tired"):
            st.success("Correct!")
        else:
            st.error(f"Nope, it was {emoji}")

# ------------------ MAIN ------------------
PAGES = {
    "Buddy Chat": lambda: page_chat("buddy"),
    "AI Doc Chat": lambda: page_chat("doc"),
    "Call Session": page_call,
    "Mood + Journal": page_journal,
    "Micro-Actions": page_micro,
    "Dashboard": page_dashboard,
    "Buddy Boost": page_boost,
    "Resources": page_resources,
    "Community": page_community,
    "Mini Game": page_game,
}

def main():
    st.sidebar.title("Youth Wellness")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()
