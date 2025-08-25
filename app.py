# app.py
"""
NYX – Youth Mental Wellness (Hackathon) — Full Upgraded App
Features:
- Buddy Chat (continuous)
- AI Doc Chat (therapy-style)
- Separate Call Session (push-to-talk) using ElevenLabs TTS (human-like) with local pyttsx3 fallback
- Mood & Journal (emoji chart, streaks, optional word cloud)
- Crisis detection on every message
- Graceful fallbacks & clear status reporting
"""

import os
import time
import tempfile
import sqlite3
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# Optional libs (guarded)
# -------------------------
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    import speech_recognition as sr
    VOICE_INPUT_AVAILABLE = True
except Exception:
    VOICE_INPUT_AVAILABLE = False

# Local TTS fallback (pyttsx3)
try:
    import pyttsx3
    TTS_LOCAL_AVAILABLE = True
    _tts_engine = pyttsx3.init()
    # attempt to pick a mac-like voice
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

# ElevenLabs TTS
ELEVEN_AVAILABLE = False
try:
    from elevenlabs import generate as eleven_generate, set_api_key as set_eleven_key
    ELEVEN_AVAILABLE = True
except Exception:
    ELEVEN_AVAILABLE = False

# OpenAI
OPENAI_AVAILABLE = False
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    OPENAI_AVAILABLE = bool(openai.api_key)
except Exception:
    OPENAI_AVAILABLE = False

# Bind ElevenLabs key if present
if ELEVEN_AVAILABLE:
    try:
        ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY") or st.secrets.get("ELEVENLABS_API_KEY", None)
        if ELEVEN_KEY:
            set_eleven_key(ELEVEN_KEY)
        else:
            ELEVEN_AVAILABLE = False
    except Exception:
        ELEVEN_AVAILABLE = False

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

# DB helpers
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            text TEXT NOT NULL,
            mood_score REAL NOT NULL,
            mood_label TEXT NOT NULL
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
# AI: talk to OpenAI properly
# -------------------------
def ai_reply_from_history(history: List[Tuple[str, str]], mode: str) -> str:
    """
    history: list of ("user"|"ai", text)
    mode: "buddy" or "doc"
    """
    if not history:
        return "I’m here. What’s coming up for you right now?"

    # If OpenAI key missing, return a friendly fallback (not the repeating therapeutic line)
    if not OPENAI_AVAILABLE:
        latest = history[-1][1]
        _, mood = analyze_mood(latest)
        return f"I hear you. {AFFIRM_TEMPLATES[mood][0]}\n\nWould you like to tell me one small detail about today?"

    system_prompt = (
        "You are a warm, empathetic teen-friendly buddy. Use short paragraphs, validate feelings, ask gentle follow-ups."
        if mode == "buddy"
        else
        "You are a supportive therapist-like AI. Use reflective listening, validation, and offer small, practical steps. Keep language non-clinical and youth-friendly."
    )

    messages = [{"role":"system","content":system_prompt}]
    # Build messages from history (limit last 16 turns)
    for speaker, text in history[-16:]:
        role = "user" if speaker == "user" else "assistant"
        messages.append({"role": role, "content": text})

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8 if mode == "buddy" else 0.75,
            max_tokens=450,
        )
        # get assistant message
        assistant_msg = resp.choices[0].message.get("content") if hasattr(resp.choices[0].message, 'get') else resp.choices[0].message.content
        if not assistant_msg:
            return "Thanks for sharing — tell me a bit more about what felt heaviest today."
        return assistant_msg.strip()
    except Exception as e:
        # show the error for debugging in UI-friendly way
        return f"(AI error: {str(e)}) I’m here — want to keep talking?"

# -------------------------
# TTS: ElevenLabs bytes + local fallback
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
    if not TTS_LOCAL_AVAILABLE:
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
  border-radius: 16px; border: 1px solid #e6e8ee;
}
.msg{max-width: 78%; padding: 10px 12px; margin: 6px 0; border-radius: 14px; line-height: 1.5; font-size: 15px;}
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
        if speaker == "user":
            st.markdown(f'<div class="msg user"><div class="bubble-name">You</div>{msg}</div>', unsafe_allow_html=True)
        else:
            klass = "doc" if mode == "doc" else "ai"
            name  = "AI Doc" if mode == "doc" else "Buddy"
            st.markdown(f'<div class="msg {klass}"><div class="bubble-name">{name}</div>{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Pages
# -------------------------
def page_mood_and_journal():
    st.subheader("📈 Mood & Journal")
    col1, col2 = st.columns([2,1])
    with col1:
        text = st.text_area("How are you feeling today?", height=140, placeholder="Type anything that’s on your mind…")
    with col2:
        if st.button("Save Check-in", use_container_width=True):
            if not text.strip():
                st.warning("Even one sentence counts ✨")
            else:
                score, label = analyze_mood(text)
                save_entry(text, score, label)
                st.success(f"Mood: {label.capitalize()} ({score:.2f})")
                st.write("**Tiny coping tip:** ", COPING_TIPS[label][int(time.time()) % len(COPING_TIPS[label])])
                st.info(f"💡 {MOTIVATIONAL_QUOTES[int(time.time()) % len(MOTIVATIONAL_QUOTES)]}")
                st.balloons()

    st.divider()
    st.subheader("Mood Timeline 🌈")
    data = fetch_entries()
    if not data:
        st.info("No entries yet. Your first check-in will appear here.")
        return

    df = {"Date": [d["ts"] for d in data], "Mood Score": [d["mood_score"] for d in data], "Mood": [d["mood_label"] for d in data]}
    emoji_map = {"positive": "🌞", "neutral": "😐", "negative": "🌧️"}
    emojis = [emoji_map[m] for m in df["Mood"]]

    fig = px.scatter(df, x="Date", y="Mood Score", color="Mood", text=emojis,
                     color_discrete_map={'positive':'#00CC96','neutral':'#636EFA','negative':'#EF553B'},
                     title="Mood Over Time")
    fig.update_traces(textposition="top center", marker=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    unique_days = {d["date"] for d in data}
    streak = get_streak(datetime.now().date(), unique_days)
    st.markdown(f"**Current streak:** {streak} day(s) 🔥")

    total = len(data)
    pos = sum(1 for d in data if d["mood_label"] == "positive")
    neu = sum(1 for d in data if d["mood_label"] == "neutral")
    neg = sum(1 for d in data if d["mood_label"] == "negative")
    st.markdown(f"**Stats:** 🌞 {pos/total*100:.0f}% | 😐 {neu/total*100:.0f}% | 🌧️ {neg/total*100:.0f}%")

    if WORDCLOUD_AVAILABLE:
        all_text = " ".join(d["text"] for d in data if d["text"])
        if all_text.strip():
            wc = WordCloud(width=640, height=320, background_color="white").generate(all_text)
            plt.figure(figsize=(8,4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
    else:
        st.caption("Word cloud: install `wordcloud` to enable.")

def page_chat(mode: str):
    title = "🗨️ Buddy Chat" if mode == "buddy" else "🧑‍⚕️ AI Doc Chat"
    st.subheader(title)
    st.caption("Looks and feels like real texting. Ongoing back-and-forth. 💬")

    key_hist = f"hist_{mode}"
    if key_hist not in st.session_state:
        st.session_state[key_hist] = []

    # Input row
    col_a, col_b, col_c = st.columns([6,1,1])
    with col_a:
        user_msg = st.text_input("Type a message…", key=f"inp_{mode}", placeholder="Tell me what's up…")
    with col_b:
        send = st.button("Send", use_container_width=True, key=f"send_{mode}")
    with col_c:
        clear = st.button("Clear", use_container_width=True, key=f"clear_{mode}")

    if clear:
        st.session_state[key_hist] = []

    # Handle send
    if send and user_msg.strip():
        st.session_state[key_hist].append(("user", user_msg))
        if detect_crisis(user_msg):
            st.error("If you’re thinking about harming yourself, reach out to trusted people or local helplines. This app isn’t a crisis service.")
        else:
            typing_indicator()
            reply = ai_reply_from_history(st.session_state[key_hist], mode)
            st.session_state[key_hist].append(("ai", reply))

    # Render bubbles
    render_chat(st.session_state[key_hist], mode)

def page_call_session():
    st.subheader("📞 Call Session (AI Doc)")
    st.caption("Press **Talk** to speak. You’ll hear a natural reply. Repeat to keep the convo going.")

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

                    # Prefer ElevenLabs (human-like), else local TTS fallback
                    played = False
                    audio_bytes = tts_elevenlabs_bytes(reply, voice=voice_name)
                    if audio_bytes:
                        # ElevenLabs returns raw bytes (mp3) — play via Streamlit
                        try:
                            st.audio(audio_bytes, format="audio/mp3")
                            played = True
                        except Exception as e:
                            st.warning(f"Could not play ElevenLabs audio in browser: {e}")
                            played = False
                    if not played:
                        if tts_local_speak(reply):
                            st.info("🎧 Played with local voice (fallback).")
                            played = True
                    if not played:
                        st.info("Voice unavailable — showing text only.")

    # Show call log like chat bubbles
    render_chat(st.session_state["hist_call"], mode="doc")

# ---- Main ----
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🫶", layout="wide")
    st.markdown(
        f"<h1 style='text-align:center'>{APP_TITLE} 🫶</h1>"
        "<p style='text-align:center;color:#667085'>A kind companion for check-ins, venting, therapy-style chats & voice calls.</p>",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🗨️ Buddy Chat",
            "🧑‍⚕️ AI Doc Chat",
            "📞 Call Session (AI Doc)",
            "📈 Mood & Journal",
        ],
    )

    with st.sidebar.expander("Status"):
        st.write("OpenAI:", "✅" if OPENAI_AVAILABLE else "⚠️ Missing key")
        st.write("ElevenLabs (TTS):", "✅" if ELEVEN_AVAILABLE else "⚠️ Missing/Not installed")
        st.write("Local TTS (pyttsx3):", "✅" if TTS_LOCAL_AVAILABLE else "⚠️ Not available")
        st.write("Mic Input:", "✅" if VOICE_INPUT_AVAILABLE else "⚠️ Install PyAudio & SpeechRecognition")
        st.caption("Add keys to `.env` or `st.secrets`")

    if page == "🗨️ Buddy Chat":
        page_chat(mode="buddy")
    elif page == "🧑‍⚕️ AI Doc Chat":
        page_chat(mode="doc")
    elif page == "📞 Call Session (AI Doc)":
        page_call_session()
    else:
        page_mood_and_journal()

if __name__ == "__main__":
    main()
