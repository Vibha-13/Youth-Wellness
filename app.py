# app.py
"""
NYX – Youth Mental Wellness (Hackathon) — All-in-one Upgraded App
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
        # If using OpenRouter, point base_url to OpenRouter endpoint
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
# AI reply function (multi-tiered)
# -------------------------
def ai_reply_from_history(history: List[Tuple[str, str]], mode: str) -> str:
    """
    Try in order:
    1) OpenAI/OpenRouter via new OpenAI client
    2) transformers conversational pipeline (DialoGPT) if installed
    3) heuristic template fallback
    """
    if not history:
        return "I’m here. What’s coming up for you right now?"

    # 1) OpenAI/OpenRouter client
    if OPENAI_AVAILABLE and openai_client:
        try:
            system_prompt = (
                "You are a warm, empathetic teen-friendly buddy. Reply in 2 short supportive paragraphs: validate, ask a gentle follow-up, and offer one small coping step. Keep language friendly and non-clinical."
                if mode == "buddy"
                else
                "You are a supportive therapist-like assistant. Reflect feelings, validate, offer 1-2 practical coping steps, and ask a gentle follow-up. Keep tone reassuring and non-clinical."
            )
            messages = [{"role":"system","content":system_prompt}]
            for speaker, text in history[-12:]:
                role = "user" if speaker == "user" else "assistant"
                messages.append({"role": role, "content": text if isinstance(text, str) else str(text)})

            model = os.getenv("OPENAI_MODEL") or os.getenv("OPENROUTER_MODEL") or "gpt-3.5-turbo"
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7 if mode == "buddy" else 0.6,
                max_tokens=512,
            )
            if resp and getattr(resp, "choices", None):
                choice = resp.choices[0]
                msg = None
                if hasattr(choice, "message"):
                    msg = getattr(choice.message, "content", None)
                elif hasattr(choice, "text"):
                    msg = choice.text
                if msg:
                    return msg.strip()
        except Exception as e:
            # log and continue to fallback
            print("OpenAI/OpenRouter call failed:", e)

    # 2) Transformers conversational fallback
    if TRANSFORMERS_AVAILABLE and conv_pipe and Conversation:
        try:
            last_user = next((t for s,t in reversed(history) if s=="user"), history[-1][1])
            conv = Conversation(last_user)
            conv_pipe(conv)
            if conv.generated_responses:
                resp_text = conv.generated_responses[-1]
                # ensure slightly longer reply if too short
                if len(resp_text.split()) < 8:
                    resp_text = resp_text + " I hear you — would you like a breathing exercise or a quick tip?"
                return resp_text
        except Exception as e:
            print("Transformers conv failed:", e)

    # 3) Heuristic/fallback
    latest = history[-1][1]
    score, mood = analyze_mood(latest)
    if mood == "positive":
        template = f"I hear you — that's wonderful to hear! {random.choice(AFFIRM_TEMPLATES['positive'])} Would you like to share more about what's working?"
    elif mood == "neutral":
        template = f"Thanks for telling me. {random.choice(AFFIRM_TEMPLATES['neutral'])} One small idea: try a quick 60-second breathing break. Want me to guide you?"
    else:
        template = f"{random.choice(AFFIRM_TEMPLATES['negative'])} I'm here with you. A small step: box breathing for one minute. Want to try it together now?"
    return template

# -------------------------
# TTS helpers
# -------------------------
def tts_elevenlabs_bytes(text: str, voice: str = "Rachel") -> Optional[bytes]:
    if not ELEVEN_AVAILABLE:
        return None
    try:
        audio_bytes = eleven_generate(text=text, voice=voice, model="eleven_multilingual_v1")
        return audio_bytes
    except Exception as e:
        st.warning(f"ElevenLabs error: {e}")
        return None

def tts_local_speak(text: str) -> bool:
    if not TTS_LOCAL_AVAILABLE or not _tts_engine:
        return False
    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
        return True
    except Exception as e:
        st.warning(f"Local TTS error: {e}")
        return False

# -------------------------
# UI / Chat rendering
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
# Guided micro-actions & visuals
# -------------------------
def guided_micro_action(label: str):
    st.markdown("### 🧘 Guided Micro-action")
    if label == "negative":
        st.write("Box Breathing: follow the prompts below (two cycles).")
        box_ph = st.empty()
        for cycle in range(2):
            for phase, secs in [("Inhale", 4), ("Hold", 4), ("Exhale", 4), ("Hold", 4)]:
                box_ph.markdown(f"<h3 style='text-align:center'>{phase} — {secs}s</h3>", unsafe_allow_html=True)
                time.sleep(secs)
        box_ph.empty()
        st.success("Nice — you did a quick reset 🌿")
    elif label == "neutral":
        gratitude = st.text_area("Gratitude journaling — write 1 thing you’re grateful for:", key=f"grat_{random.randint(1,9999)}")
        if st.button("Save gratitude"):
            if gratitude.strip():
                save_journal("GRATITUDE: " + gratitude.strip())
                st.success("Saved — small gratitude, big effect ✨")
            else:
                st.warning("Write at least one line!")
    else:
        st.write("Celebrate a small win — type one tiny win below:")
        win = st.text_input("Tiny win")
        if st.button("Save tiny win"):
            if win.strip():
                save_journal("WIN: " + win.strip())
                st.success("Saved — keep stacking wins! 🏆")
            else:
                st.warning("Type something small — it counts!")

def mood_insights_dashboard(entries):
    st.subheader("📊 Mood Insights")
    if not entries:
        st.info("No check-ins yet.")
        return

    df = pd.DataFrame(entries)
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"])
    mood_counts = df["mood_label"].value_counts().reset_index()
    mood_counts.columns = ["mood", "count"]

    fig = px.pie(mood_counts, values="count", names="mood", title="Mood Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # top triggers
    all_text = " ".join(df["text"].astype(str).tolist())
    words = [w.lower().strip(".,!?:;()[]") for w in all_text.split() if len(w) > 2]
    stop = set(["that","this","have","with","your","you","just","like","and","the","for","but","not","are","was","it"])
    filtered = [w for w in words if w not in stop]
    common = pd.Series(filtered).value_counts().head(6)
    st.write("**Top triggers / words in entries:** ", ", ".join(common.index.tolist()))

    # weekday table
    df["weekday"] = df["ts"].dt.day_name()
    heat = pd.crosstab(df["weekday"], df["mood_label"])
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = heat.reindex([d for d in weekdays if d in heat.index])
    st.write("Mood counts by weekday:")
    st.dataframe(heat)

    if WORDCLOUD_AVAILABLE:
        wc_text = " ".join(df["text"].astype(str).tolist())
        if wc_text.strip():
            wc = WordCloud(width=640, height=320, background_color="white").generate(wc_text)
            plt.figure(figsize=(8,3.5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
    else:
        st.caption("Install `wordcloud` to enable a word cloud visualization.")

# Buddy Boost (downloadable card via matplotlib)
def generate_buddy_card(text: str, author: str = "Your Buddy") -> bytes:
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.axis("off")
    if NP_AVAILABLE:
        grad = np.linspace(0,1,256)
        grad = np.tile(grad,(256,1))
        ax.imshow(grad, cmap="plasma", aspect="auto", extent=(0,1,0,1))
    else:
        ax.set_facecolor("#ff7f50")
    ax.text(0.5, 0.6, text, fontsize=20, ha="center", va="center", weight="bold", color="white")
    ax.text(0.5, 0.22, f"- {author}", fontsize=12, ha="center", va="center", color="white", alpha=0.9)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# -------------------------
# Pages
# -------------------------
def page_mood_and_journal():
    st.subheader("📈 Mood & Journal")
    col1, col2 = st.columns([2,1])
    with col1:
        text = st.text_area("How are you feeling today? (One sentence is enough)", height=140, placeholder="Type anything that’s on your mind…")
    with col2:
        if st.button("Save Check-in", use_container_width=True):
            if not text.strip():
                st.warning("Even one sentence counts ✨")
            else:
                score, label = analyze_mood(text)
                if not st.session_state.get("private_mode", False):
                    save_entry(text, score, label)
                st.success(f"Mood: {label.capitalize()} ({score:.2f})")
                st.write("**Tiny coping tip:** ", random.choice(COPING_TIPS[label]))
                st.info(f"💡 {random.choice(MOTIVATIONAL_QUOTES)}")
                st.balloons()
                guided_micro_action(label)

    st.divider()
    st.subheader("Mood Timeline 🌈")
    data = fetch_entries()
    if not data:
        st.info("No entries yet. Your first check-in will appear here.")
    else:
        df = {"Date": [d["ts"] for d in data], "Mood Score": [d["mood_score"] for d in data], "Mood": [d["mood_label"] for d in data]}
        emoji_map = {"positive": "🌞", "neutral": "😐", "negative": "🌧️"}
        emojis = [emoji_map[m] for m in df["Mood"]]

        fig = px.scatter(pd.DataFrame(df), x="Date", y="Mood Score", color="Mood", text=emojis,
                         color_discrete_map={'positive':'#00CC96','neutral':'#636EFA','negative':'#EF553B'},
                         title="Mood Over Time")
        fig.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

        unique_days = {d["date"] for d in data}
        streak = get_streak(datetime.now().date(), unique_days)
        st.markdown(f"**Current streak:** {streak} day(s) 🔥")

        total = len(data)
        pos = sum(1 for d in data if d["mood_label"] == "positive")
        neu = sum(1 for d in data if d["mood_label"] == "neutral")
        neg = sum(1 for d in data if d["mood_label"] == "negative")
        st.markdown(f"**Stats:** 🌞 {pos/total*100:.0f}% | 😐 {neu/total*100:.0f}% | 🌧️ {neg/total*100:.0f}%")

        st.divider()
        mood_insights_dashboard(data)

    # Journal quick-save
    st.divider()
    st.subheader("📓 Journal")
    journal_text = st.text_area("Write a private journal entry (saved locally)", height=120, key="journal_text")
    if st.button("Save Journal Entry"):
        if journal_text.strip():
            save_journal(journal_text.strip())
            st.success("Journal saved locally ✅")
        else:
            st.warning("Write something to save!")

    # Buddy boost card
    st.divider()
    st.subheader("💌 Buddy Boost — share a tiny card")
    quote = st.selectbox("Choose a quote", MOTIVATIONAL_QUOTES)
    card_author = st.text_input("Who should sign it? (optional)", value="Your Buddy")
    if st.button("Generate & Download Card"):
        try:
            img_bytes = generate_buddy_card(quote, card_author or "Your Buddy")
            st.download_button("Download motivational card (PNG)", data=img_bytes, file_name="buddy_boost.png", mime="image/png")
            st.success("Generated — download ready!")
        except Exception as e:
            st.error(f"Could not generate card: {e}")
            st.download_button("Download quote as text", data=quote.encode(), file_name="buddy_boost.txt")

def page_chat(mode: str):
    title = "🗨️ Buddy Chat" if mode == "buddy" else "🧑‍⚕️ AI Doc Chat"
    st.subheader(title)
    st.caption("Ongoing back-and-forth. Keep the conversation flowing 💬")

    key_hist = f"hist_{mode}"
    if key_hist not in st.session_state:
        st.session_state[key_hist] = []

    col_a, col_b, col_c = st.columns([6,1,1])
    with col_a:
        user_msg = st.text_input("Type a message…", key=f"inp_{mode}", placeholder="Tell me what's up…")
    with col_b:
        send = st.button("Send", use_container_width=True, key=f"send_{mode}")
    with col_c:
        clear = st.button("Clear", use_container_width=True, key=f"clear_{mode}")

    if clear:
        st.session_state[key_hist] = []

    if send and user_msg.strip():
        st.session_state[key_hist].append(("user", user_msg))
        if detect_crisis(user_msg):
            st.error("If you’re thinking about harming yourself, reach out to trusted people or local helplines. This app isn’t a crisis service.")
            st.markdown("**Crisis resources:** If you are in immediate danger, call your local emergency number or a crisis hotline.")
        else:
            typing_indicator()
            reply = ai_reply_from_history(st.session_state[key_hist], mode)
            st.session_state[key_hist].append(("ai", reply))

    render_chat(st.session_state[key_hist], mode)

def page_call_session():
    st.subheader("📞 Call Session (AI Doc)")
    st.caption("Press **Talk** to speak. You’ll hear a natural reply (if TTS available).")

    if "hist_call" not in st.session_state:
        st.session_state["hist_call"] = []

    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        talk = st.button("🎤 Talk", use_container_width=True)
    with col2:
        voice_name = st.selectbox("Voice", ["Rachel", "Bella", "Antoni", "Adam", "Domi"], index=0)
    with col3:
        clear = st.button("Clear Log", use_container_width=True)

    if clear:
        st.session_state["hist_call"] = []

    if talk:
        if not VOICE_INPUT_AVAILABLE:
            st.error("Mic not available. Install `pyaudio` & `SpeechRecognition` to enable voice input.")
        else:
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    st.toast("🎤 Listening…", icon="🎤")
                    r.adjust_for_ambient_noise(source, duration=0.6)
                    audio = r.listen(source, timeout=6, phrase_time_limit=18)
                try:
                    utterance = r.recognize_google(audio)
                except Exception:
                    utterance = ""
            except Exception:
                utterance = ""

            if not utterance:
                st.warning("I couldn’t catch that — try again closer to the mic.")
            else:
                st.session_state["hist_call"].append(("user", utterance))
                if detect_crisis(utterance):
                    st.error("If you’re thinking about harming yourself, reach out to trusted people or local helplines. This app isn’t a crisis service.")
                else:
                    typing_indicator()
                    reply = ai_reply_from_history(st.session_state["hist_call"], mode="doc")
                    st.session_state["hist_call"].append(("ai", reply))

                    played = False
                    audio_bytes = tts_elevenlabs_bytes(reply, voice=voice_name)
                    if audio_bytes:
                        try:
                            st.audio(audio_bytes, format="audio/mp3")
                            played = True
                        except Exception as e:
                            st.warning(f"Could not play ElevenLabs audio in browser: {e}")
                            played = False
                    if not played and tts_local_speak(reply):
                        st.info("🎧 Played with local voice (fallback).")
                        played = True
                    if not played:
                        st.info("Voice unavailable — showing text only.")

    render_chat(st.session_state["hist_call"], mode="doc")

def page_game():
    st.subheader("🎮 Mini Stress Relief — Guess the Number")
    if "secret" not in st.session_state:
        st.session_state.secret = random.randint(1,20)
    guess = st.number_input("Guess a number 1-20", 1, 20)
    if st.button("Check"):
        if guess == st.session_state.secret:
            st.balloons()
            st.success("You guessed it! 🎉")
            st.session_state.secret = random.randint(1,20)
        else:
            st.warning("Try again!")

def page_journal_and_community():
    st.subheader("📓 Journaling")
    entry = st.text_area("Write anything (journal will be saved locally)", height=120)
    if st.button("Save Journal"):
        if entry.strip():
            save_journal(entry.strip())
            st.success("Saved locally ✅")
        else:
            st.warning("Write something to save!")

    st.divider()
    st.subheader("💬 Community Chat")
    name = st.text_input("Your display name (optional)", key="comm_name")
    msg = st.text_input("Message", key="comm_msg")
    if st.button("Send Message"):
        if msg.strip():
            save_community(name.strip() or "Anonymous", msg.strip())
            st.success("Message posted")
        else:
            st.warning("Write a message to post!")
    st.write("Recent messages:")
    for ts, nm, m in fetch_community(20):
        t = datetime.fromisoformat(ts).strftime("%b %d %H:%M")
        st.markdown(f"**{nm}** · *{t}* — {m}")

def page_doctor_call_and_booking():
    st.subheader("👩‍⚕️ Doctor Call & Free Therapy Booking")
    st.write("If you need real professional help, please use licensed telehealth providers. This link opens a meeting (demo).")
    st.markdown("[Start a video call (demo)](https://meet.google.com/)", unsafe_allow_html=True)

    st.divider()
    st.subheader("🆓 Free Therapy Booking (demo)")
    name = st.text_input("Your name for booking")
    d = st.date_input("Pick a date", min_value=date.today())
    t = st.time_input("Pick a time", value=dtime(hour=10, minute=0))
    if st.button("Book Free Session"):
        if not name.strip():
            st.warning("Please enter your name to book.")
        else:
            save_booking(name.strip(), d, t)
            st.success(f"Booked {name} on {d.isoformat()} at {t.isoformat()}. (Demo)")

    st.write("Recent bookings (demo):")
    for ts, nm, bdate, btime in fetch_bookings(10):
        st.write(f"- {nm} — {bdate} {btime}")

# Crisis panel
def crisis_resources():
    with st.sidebar.expander("🚨 Crisis Resources", expanded=True):
        st.write("If you're in immediate danger, call your local emergency number.")
        st.write("- India (example): 112")
        st.write("- If you are thinking about self-harm: talk to someone you trust now.")
        st.write("Grounding exercise: 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.")
        st.write("This app is not a substitute for professional care. See Doctor Call tab to connect to a professional service.")

# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🫶", layout="wide")
    st.markdown(
        f"<h1 style='text-align:center'>{APP_TITLE} 🫶</h1>"
        "<p style='text-align:center;color:#667085'>A kind companion for check-ins, venting, therapy-style chats & voice calls.</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.title("NYX Controls")
    st.sidebar.markdown("**Status**")
    st.sidebar.write("OpenRouter/OpenAI configured:", "✅" if OPENAI_AVAILABLE else "⚠️ Not configured")
    st.sidebar.write("Transformers conv:", "✅" if TRANSFORMERS_AVAILABLE else "⚠️ Not installed")
    st.sidebar.write("ElevenLabs (TTS):", "✅" if ELEVEN_AVAILABLE else "⚠️ Missing/Not installed")
    st.sidebar.write("Local TTS (pyttsx3):", "✅" if TTS_LOCAL_AVAILABLE else "⚠️ Not available")
    st.sidebar.write("Mic Input:", "✅" if VOICE_INPUT_AVAILABLE else "⚠️ Install PyAudio & SpeechRecognition")
    st.sidebar.caption("Put keys into .env or st.secrets (OPENROUTER_API_KEY or OPENAI_API_KEY / ELEVENLABS_API_KEY)")

    if "private_mode" not in st.session_state:
        st.session_state["private_mode"] = False
    st.session_state["private_mode"] = st.sidebar.checkbox("Private Mode (don't save check-ins)", value=st.session_state["private_mode"])
    if st.sidebar.button("Clear All Local Data"):
        conn.execute("DELETE FROM entries"); conn.execute("DELETE FROM journal"); conn.execute("DELETE FROM community"); conn.execute("DELETE FROM bookings"); conn.commit()
        st.sidebar.success("All local data cleared")

    crisis_resources()

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
