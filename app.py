# app.py
"""
Upgraded AI Wellness Companion
- Fixed streaks, safe DB guards, browser TTS, PDF export fallback, cleaned text for AI
"""

import streamlit as st
import os
import time
import random
import io
import math
import re
import json
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# NLP & visuals
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Audio libs (optional)
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    import wavio
except Exception:
    wavio = None

# Local TTS (optional)
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Supabase (optional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Browser components (for TTS fallback)
import streamlit.components.v1 as components

# Optional PDF generator
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas
except Exception:
    pdf_canvas = None

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# ---------- STYLES ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; font-family: 'Poppins', sans-serif; }
    .main .block-container { padding: 2rem 4rem; }
    .card { background-color: #eaf4ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 25px; margin-bottom: 25px; border-left: 5px solid #4a90e2; transition: transform .18s; }
    .card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    .stButton>button { color: #fff; background-color: #4a90e2; border-radius: 8px; padding: 10px 22px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SERVICES ----------
GEMINI_API_KEY = None
if st.secrets:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or GEMINI_API_KEY
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_available = False
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        ai_available = True
    except Exception:
        ai_available = False
        st.sidebar.warning("AI API connection failed — falling back to local responses.")
else:
    st.sidebar.info("AI: Local fallback mode (no GEMINI key).")

SUPABASE_URL = None
SUPABASE_KEY = None
if st.secrets:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL") or SUPABASE_URL
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or SUPABASE_KEY
if not SUPABASE_URL:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_KEY:
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
db_connected = False
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        db_connected = True
    except Exception:
        db_connected = False

st.sidebar.markdown(f"- AI: **{'Connected' if ai_available else 'Local (fallback)'}**")
st.sidebar.markdown(f"- DB: **{'Connected' if db_connected else 'Not connected'}**")

# ---------- STATE ----------
if "messages" not in st.session_state: st.session_state["messages"] = []
if "call_history" not in st.session_state: st.session_state["call_history"] = []
if "daily_journal" not in st.session_state: st.session_state["daily_journal"] = []
if "mood_history" not in st.session_state: st.session_state["mood_history"] = []
if "streaks" not in st.session_state:
    st.session_state["streaks"] = {"mood_log": 0, "last_mood_date": None, "badges": []}
if "transcription_text" not in st.session_state: st.session_state["transcription_text"] = ""
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "user_email" not in st.session_state: st.session_state["user_email"] = None

analyzer = SentimentIntensityAnalyzer()

# ---------- CONTENT ----------
QUOTES = [
    "You are stronger than you think. 💪",
    "Even small steps count. 🌱",
    "Breathe. You are doing your best. 🌬️",
    "This moment will pass. You're doing important work by being here. 💛",
    "Progress, not perfection. Tiny steps add up."
]

MOOD_EMOJI_MAP = {1:"😭",2:"😢",3:"😔",4:"😕",5:"😐",6:"🙂",7:"😊",8:"😄",9:"🤩",10:"🥳"}
BADGE_RULES = [
    ("Getting Started", lambda s: len(s["mood_history"]) >= 1),
    ("Weekly Streak: 3", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 3),
    ("Consistent 7", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 7),
]

# ---------- HELPERS ----------
def now_ts(): return time.time()

def clean_text_for_ai(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)   # strip non-ascii (emojis)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300) -> str:
    prompt_clean = clean_text_for_ai(prompt)
    if ai_available:
        try:
            resp = model.generate_content(prompt_clean)
            text = getattr(resp, "text", None) or str(resp)
            return text
        except Exception:
            st.warning("AI generation failed — using fallback response.")
    canned = [
        "Thanks for sharing. I hear you — would you like to tell me more?",
        "That’s a lot to carry. I’m here. Could you describe one small thing that feels heavy right now?",
        "I’m listening. If you want, we can try a 1-minute breathing exercise together."
    ]
    return random.choice(canned)

def sentiment_compound(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"]

def get_all_user_text() -> str:
    parts = []
    parts += [e.get("text","") for e in st.session_state["daily_journal"] if e.get("text")]
    parts += [m.get("content","") for m in st.session_state["messages"] if m.get("role") == "user" and m.get("content")]
    parts += [c.get("text","") for c in st.session_state["call_history"] if c.get("speaker") == "User" and c.get("text")]
    return " ".join(parts).strip()

def generate_wordcloud_figure(text: str):
    if not text or not text.strip(): return None
    try:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception as e:
        st.warning(f"WordCloud failed: {e}")
        return None

def record_audio(duration=5, fs=44100):
    if sd is None:
        st.warning("Recording not available in this environment.")
        return None
    st.info("Recording... speak now.")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True, dtype='int16')
    st.success("Recording complete.")
    if wavio:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wavio.write(tmp.name, audio_data, fs, sampwidth=2)
            tmp.close()
            with open(tmp.name, "rb") as f:
                b = io.BytesIO(f.read())
            return b
        except Exception as e:
            st.warning(f"Could not write WAV: {e}")
            return None
    else:
        st.warning("wavio not installed — returning None for audio file.")
        return None

# Browser TTS first, then pyttsx3 local fallback
def browser_tts(text: str) -> bool:
    try:
        payload = json.dumps({"text": text})
        components.html(f"""
            <script>
            const payload = {payload};
            const utter = new SpeechSynthesisUtterance(payload.text);
            utter.rate = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utter);
            </script>
        """, height=0)
        return True
    except Exception:
        return False

def speak_text(text: str):
    if not text: return
    if browser_tts(text):
        return
    if pyttsx3:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            st.warning("Local TTS failed.")

# ---------- Supabase helpers (guarded) ----------
def register_user_db(email: str):
    if not db_connected or supabase is None: return None
    try:
        res = supabase.table("users").insert({"email": email}).execute()
        if getattr(res, "data", None):
            return res.data[0].get("id")
    except Exception as e:
        st.warning(f"Supabase register failed: {e}")
    return None

def get_user_by_email_db(email: str):
    if not db_connected or supabase is None: return []
    try:
        res = supabase.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    if not db_connected or supabase is None: return False
    try:
        supabase.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception as e:
        st.warning(f"Save to DB failed: {e}")
        return False

def load_journal_db(user_id):
    if not db_connected or supabase is None: return []
    try:
        res = supabase.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

# ---------- UI pieces ----------
def sidebar_auth():
    st.sidebar.header("Account")
    if not st.session_state.get("logged_in"):
        email = st.sidebar.text_input("Your email", key="login_email")
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                if db_connected:
                    user = get_user_by_email_db(email)
                if user:
                    st.session_state["user_id"] = user[0].get("id")
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    entries = load_journal_db(st.session_state["user_id"]) or []
                    st.session_state["daily_journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in entries]
                    st.sidebar.success("Logged in.")
                    st.rerun()
                else:
                    uid = None
                    if db_connected:
                        uid = register_user_db(email)
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.sidebar.success("Registered & logged in.")
                        st.rerun()
                    else:
                        st.session_state["logged_in"] = True
                        st.session_state["user_email"] = email
                        st.sidebar.info("Logged in locally (no DB).")
                        st.rerun()
            else:
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user_id"] = None
            st.session_state["user_email"] = None
            st.session_state["daily_journal"] = []
            st.sidebar.info("Logged out.")
            st.rerun()

# ---------- App pages ----------
def homepage_panel():
    st.title("Your Wellness Sanctuary")
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("Daily Inspiration")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{random.choice(QUOTES)}**")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("### Quick actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Start Breathing"):
                st.session_state["page"] = "breathing"
                st.rerun()
        with c2:
            if st.button("Talk to AI"):
                st.session_state["page"] = "chat"
                st.rerun()
        with c3:
            if st.button("Journal"):
                st.session_state["page"] = "journaling"
                st.rerun()
    with col2:
        st.image("https://images.unsplash.com/photo-1549490349-f06b3e942007?q=80&w=2070&auto=format&fit=crop", caption="Take a moment for yourself")
    st.markdown("---")
    st.header("Features")
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown("#### Mood Tracker")
        st.markdown("Log quick mood ratings and unlock badges.")
    with f2:
        st.markdown("#### AI Doc Chat")
        st.markdown("A compassionate AI to listen and suggest small exercises.")
    with f3:
        st.markdown("#### Journal & Insights")
        st.markdown("Track progress over time with charts and word clouds.")

def mood_tracker_panel():
    st.header("Daily Mood Tracker")
    col1, col2 = st.columns([3,1])
    with col1:
        mood = st.slider("How do you feel right now?", 1, 10, 6)
        st.markdown(f"**You chose:** {MOOD_EMOJI_MAP[mood]} · {mood}/10")
        note = st.text_input("Optional: Add a short note about why you feel this way")
        if st.button("Log Mood"):
            entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
            st.session_state["mood_history"].append(entry)

            # update streaks safely
            last_date = st.session_state["streaks"].get("last_mood_date")
            today = datetime.now().date()
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
                except Exception:
                    last_dt = None
            else:
                last_dt = None

            if last_dt == today:
                pass
            else:
                yesterday = today - timedelta(days=1)
                if last_dt == yesterday:
                    st.session_state["streaks"]["mood_log"] = st.session_state["streaks"].get("mood_log", 0) + 1
                else:
                    st.session_state["streaks"]["mood_log"] = 1
                st.session_state["streaks"]["last_mood_date"] = today.strftime("%Y-%m-%d")

            st.success("Mood logged. Tiny step, big impact.")

            # badges
            for name, rule in BADGE_RULES:
                try:
                    if rule({"mood_history": st.session_state["mood_history"], "streaks": st.session_state["streaks"]}):
                        if name not in st.session_state["streaks"]["badges"]:
                            st.session_state["streaks"]["badges"].append(name)
                except Exception:
                    continue
            st.rerun()

    with col2:
        st.subheader("Badges")
        for b in st.session_state["streaks"]["badges"]:
            st.markdown(f"- 🏅 {b}")
        st.subheader("Streak")
        st.markdown(f"Consecutive days logging mood: **{st.session_state['streaks'].get('mood_log',0)}**")

    if st.session_state["mood_history"]:
        df = pd.DataFrame(st.session_state["mood_history"]).copy()
        df['date'] = pd.to_datetime(df['date'])
        fig = px.line(df, x='date', y='mood', title="Mood Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def ai_doc_chat_panel():
    st.header("AI Doc Chat")
    st.markdown("A compassionate AI buddy to listen.")
    for message in st.session_state["messages"]:
        role = message.get("role", "user")
        try:
            with st.chat_message(role):
                st.markdown(message.get("content", ""))
        except Exception:
            if role == "user":
                st.markdown(f"**You:** {message.get('content','')}")
            else:
                st.markdown(f"**AI:** {message.get('content','')}")
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state["messages"].append({"role":"user","content":prompt,"ts":now_ts()})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_context = "\n\n".join([m["content"] for m in st.session_state["messages"][-6:]])
                ai_resp = safe_generate(prompt_context)
                st.markdown(ai_resp)
                st.session_state["messages"].append({"role":"assistant","content":ai_resp,"ts":now_ts()})
        st.rerun()

def call_session_panel():
    st.header("Call Session (Record & Reply)")
    st.markdown("Record a short message — the app will transcribe (demo) and reply.")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Recording duration (seconds)", 3, 20, 8)
        if sd is None:
            st.info("Recording disabled on this environment.")
        if st.button("Start Recording"):
            audio = record_audio(duration=duration)
            if audio:
                trans = "This is a short transcription placeholder. You can replace this with a real STT when available."
                st.session_state["transcription_text"] = trans
                st.session_state["call_history"].append({"speaker":"User","text":trans,"timestamp":now_ts()})
                st.rerun()
    if st.session_state.get("transcription_text"):
        st.subheader("You said:")
        st.info(st.session_state["transcription_text"])
        if st.button("Get AI Reply"):
            st.session_state["messages"].append({"role":"user","content":st.session_state["transcription_text"],"ts":now_ts()})
            ai_resp = safe_generate(st.session_state["transcription_text"])
            st.session_state["call_history"].append({"speaker":"AI","text":ai_resp,"timestamp":now_ts()})
            st.subheader("AI Reply:")
            st.markdown(ai_resp)
            try:
                speak_text(ai_resp)
            except Exception:
                st.warning("TTS not available in this environment.")
            st.session_state["transcription_text"] = ""
            st.rerun()
    if st.session_state["call_history"]:
        st.markdown("---")
        st.subheader("Call History")
        for e in st.session_state["call_history"][-10:]:
            who = "user" if e["speaker"] == "User" else "assistant"
            try:
                with st.chat_message(who):
                    st.markdown(e.get("text",""))
            except Exception:
                st.markdown(f"**{e.get('speaker')}:** {e.get('text')}")

def mindful_breathing_panel():
    st.header("Mindful Breathing — 4-4-6 (Short)")
    st.markdown("Follow the prompts: Inhale (4s) — Hold (4s) — Exhale (6s). Try 3 cycles.")
    if "breath_running" not in st.session_state:
        st.session_state.breath_running = False
        st.session_state.breath_phase = ""
        st.session_state.breath_cycle = 0
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Start Exercise"):
            st.session_state.breath_running = True
            st.session_state.breath_cycle = 0
            st.rerun()
    with c2:
        if st.button("Reset"):
            st.session_state.breath_running = False
            st.session_state.breath_phase = ""
            st.session_state.breath_cycle = 0
            st.rerun()
    if st.session_state.breath_running:
        cycles = 3
        pattern = [("Inhale",4),("Hold",4),("Exhale",6)]
        for c in range(st.session_state.breath_cycle, cycles):
            for phase, sec in pattern:
                st.session_state.breath_phase = phase
                st.markdown(f"**{phase}** — {sec} seconds")
                placeholder = st.empty()
                for t in range(sec,0,-1):
                    placeholder.markdown(f"<h2 style='color:#374151'>{t}</h2>", unsafe_allow_html=True)
                    time.sleep(1)
                placeholder.empty()
            st.session_state.breath_cycle = c+1
            st.rerun()
        st.session_state.breath_running = False
        st.balloons()
        st.success("Nice job — that was mindful breathing!")

def mindful_journaling_panel():
    st.header("Mindful Journaling")
    st.markdown("Write freely — your words are private here unless you save to your account.")
    journal_text = st.text_area("Today's reflection", height=220, key="journal_text")
    if st.button("Save Entry"):
        if journal_text.strip():
            sent = sentiment_compound(journal_text)
            if st.session_state.get("logged_in") and db_connected and st.session_state.get("user_id"):
                ok = save_journal_db(st.session_state.get("user_id"), journal_text, sent)
                if ok:
                    st.success("Saved to your account.")
                else:
                    st.warning("Could not save to DB. Saved locally instead.")
                    st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
            else:
                st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
                st.success("Saved locally.")
            st.rerun()
        else:
            st.warning("Write something you want to save.")

def journal_analysis_panel():
    st.header("Journal & Analysis")
    all_text = get_all_user_text()
    if not all_text:
        st.info("No journal or call text yet — start journaling or talking to get insights.")
        return
    entries = []
    for e in st.session_state["daily_journal"]:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e.get("sentiment",0)})
    for ch in st.session_state["call_history"]:
        if ch.get("speaker") == "User":
            entries.append({"date": pd.to_datetime(datetime.fromtimestamp(ch["timestamp"])), "compound": sentiment_compound(ch.get("text",""))})
    if entries:
        df = pd.DataFrame(entries).sort_values("date")
        df["sentiment_label"] = df["compound"].apply(lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
        fig = px.line(df, x="date", y="compound", color="sentiment_label", markers=True, title="Sentiment Over Time", color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"})
        st.plotly_chart(fig, use_container_width=True)
    wc_fig = generate_wordcloud_figure(all_text)
    if wc_fig:
        st.subheader("Word Cloud")
        st.pyplot(wc_fig, clear_figure=True)

def mini_quiz_panel():
    st.header("Mood Booster — Quick Quiz")
    st.markdown("Try this 3-question quiz. It’s a fun mood booster — not a medical test.")
    questions = [
        {"q":"Pick a color you like:", "a":["Blue","Green","Purple","Yellow"]},
        {"q":"Pick a coping micro-tool:", "a":["Deep breath","Short walk","Music","Call a friend"]},
        {"q":"Choose a calming sound:", "a":["Rain","Ocean","Wind chimes","Silence"]}
    ]
    answers = []
    for i, item in enumerate(questions):
        ans = st.radio(item["q"], item["a"], key=f"q{i}")
        answers.append(ans)
    if st.button("Get Suggestion"):
        suggestion = safe_generate("User choices: " + ", ".join(answers))
        st.subheader("Suggestion:")
        st.info(suggestion)
        if "Mood Booster" not in st.session_state["streaks"]["badges"]:
            st.session_state["streaks"]["badges"].append("Mood Booster")
        st.rerun()

def emotional_journey_panel():
    st.header("My Emotional Journey")
    all_text = get_all_user_text()
    if not all_text:
        st.info("Interact with the app more to build an emotional journey.")
        return
    st.subheader("AI-generated narrative (empathetic)")
    prompt = f\"\"\"\nWrite a short, supportive, and strength-focused 3-paragraph narrative about a person's recent emotional journey.\nUse empathetic tone and offer gentle encouragement. Data:\n{clean_text_for_ai(all_text)[:4000]}\n\"\"\"\n    if ai_available:\n        try:\n            story = model.generate_content(prompt).text\n            st.markdown(story)\n            return\n        except Exception:\n            st.warning(\"AI generation failed; showing fallback.\")\n    fallback_story = \"You’ve been carrying a lot — and showing up to this app is a small brave step. Over time, small acts of care add up. Keep logging your moments and celebrate tiny wins.\"\n    st.markdown(fallback_story)

def personalized_report_panel():
    st.header("Personalized Report")
    all_text = get_all_user_text()
    if not all_text:
        st.info("No data yet. Start journaling or chatting to generate a report.")
        return
    entries = []
    for e in st.session_state["daily_journal"]:
        entries.append(e)
    for ch in st.session_state["call_history"]:
        if ch.get("speaker") == "User":
            entries.append({"date": datetime.fromtimestamp(ch.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S"), "text": ch.get("text"), "sentiment": sentiment_compound(ch.get("text",""))})
    df = pd.DataFrame(entries)
    pos = len(df[df.get("sentiment",0) >= 0.05]) if not df.empty else 0
    neg = len(df[df.get("sentiment",0) <= -0.05]) if not df.empty else 0
    neut = len(df) - pos - neg if not df.empty else 0
    st.subheader("Analysis Summary")
    st.markdown(f"**Entries analyzed:** {len(df)}")
    st.markdown(f"- Positive: {pos}")
    st.markdown(f"- Neutral: {neut}")
    st.markdown(f"- Negative: {neg}")
    insight_prompt = f"Summarize the main emotional themes in these notes and give 3 gentle suggestions: {clean_text_for_ai(all_text)[:4000]}"
    insight = safe_generate(insight_prompt)
    st.subheader("AI Insight")
    st.write(insight)
    report_text = f"Summary generated on {datetime.now().strftime('%Y-%m-%d')}\nEntries: {len(df)}\nPositive:{pos}\nNeutral:{neut}\nNegative:{neg}\n\nAI Insight:\n{insight}\n\nRaw text:\n{all_text}"
    if pdf_canvas:
        try:
            mem = io.BytesIO()
            c = pdf_canvas(mem, pagesize=letter)
            width, height = letter
            y = height - 40
            c.setFont('Helvetica-Bold', 14)
            c.drawString(40, y, 'Personalized Wellness Report')
            y -= 30
            c.setFont('Helvetica', 10)
            for line in report_text.split('\n'):
                if y < 60:
                    c.showPage()
                    y = height - 40
                    c.setFont('Helvetica', 10)
                c.drawString(40, y, line[:120])
                y -= 14
            c.save()
            mem.seek(0)
            st.download_button('Download Report (PDF)', data=mem, file_name='wellness_report.pdf', mime='application/pdf')
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")
            st.download_button('Download Report (TXT)', data=report_text, file_name='wellness_report.txt', mime='text/plain')
    else:
        st.download_button('Download Report (TXT)', data=report_text, file_name='wellness_report.txt', mime='text/plain')

def crisis_support_panel():
    st.header("Crisis Support — Immediate Resources")
    st.markdown("If you are in immediate danger or suicidal, please contact local emergency services now.")
    st.markdown("**Helplines (US examples)**: 988 (US Suicide & Crisis Lifeline). Replace with local hotlines if outside US.")
    st.markdown("**Grounding exercise (5-4-3-2-1)** — try naming: 5 things you see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.")
    if st.button("Quick grounding"):
        st.info("Look around and name 5 things you see right now.")
        time.sleep(0.5)

# ---------- NAVIGATION ----------
def main():
    st.sidebar.title("Navigation")
    sidebar_auth()
    pages = {
        "Home": homepage_panel,
        "Mood Tracker": mood_tracker_panel,
        "AI Doc Chat": ai_doc_chat_panel,
        "Call Session": call_session_panel,
        "Mindful Breathing": mindful_breathing_panel,
        "Mindful Journaling": mindful_journaling_panel,
        "Journal & Analysis": journal_analysis_panel,
        "Mini Quiz": mini_quiz_panel,
        "My Emotional Journey": emotional_journey_panel,
        "Personalized Report": personalized_report_panel,
        "Crisis Support": crisis_support_panel
    }
    page = st.sidebar.radio("Go to:", list(pages.keys()), index=0)
    func = pages.get(page)
    if func:
        func()
    st.markdown("---")
    st.markdown("Built with care • Data stored locally unless you log in and save to your account.")

if __name__ == "__main__":
    main()
