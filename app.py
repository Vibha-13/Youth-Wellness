"""
AI Wellness Companion
A full-featured Streamlit app for mental wellness, including:
- Mood Tracker
- AI Chat
- Journaling & Analysis
- Mindful Breathing
- User Authentication
- Data persistence with Supabase (optional)
- AI integration with Google Gemini (optional)
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

# Audio libs
try:
    from pydub import AudioSegment
except ImportError:
    st.warning("Pydub not installed. Some audio functionality may not work.")
    AudioSegment = None

try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
except ImportError:
    st.warning("Streamlit-webrtc not installed. The call session panel will not work.")
    webrtc_streamer = None
    AudioProcessorBase = None # ADDED THIS LINE

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

# ---------- CONFIG & SETUP ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# Set up session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# ---------- STYLES ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; font-family: 'Poppins', sans-serif; }
    .main .block-container { padding: 2rem 4rem; }
    .card { background-color: #eaf4ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 25px; margin-bottom: 25px; border-left: 5px solid #4a90e2; transition: transform .18s; }
    .card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    .stButton>button { color: #fff; background-color: #4a90e2; border-radius: 8px; padding: 10px 22px; font-weight:600; border: none; }
    .stButton>button:hover { background-color: #357bd9; }
    .st-emotion-cache-1av55r7 {
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
        padding: 35px;
        border: none;
    }
    .st-emotion-cache-16p649c {
        border: none;
        border-radius: 15px;
        background-color: #f0f4f8;
        padding: 20px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-12oz5g7 {
        background-color: #eaf4ff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4a90e2;
    }
    .st-emotion-cache-12oz5g7:hover {
        background-color: #dbeaff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SERVICES ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

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

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

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
if "phq9_score" not in st.session_state: st.session_state.phq9_score = None
if "phq9_interpretation" not in st.session_state: st.session_state.phq9_interpretation = None

analyzer = SentimentIntensityAnalyzer()

# ---------- CONTENT ----------
QUOTES = [
    "You are stronger than you think. 💪",
    "Even small steps count. 🌱",
    "Breathe. You are doing your best. 🌬️",
    "This moment will pass. You're doing important work by being here. 💛",
    "Progress, not perfection. Tiny steps add up."
]

MOOD_EMOJI_MAP = {1:"😭",2:"😢",3:"😔",4:"😕",5:"😐",6:"🙂",7:"😊",8:"😄",9:"🤩",10:"🥳", 11: "✨"}
BADGE_RULES = [
    ("Getting Started", lambda s: len(s["mood_history"]) >= 1),
    ("Weekly Streak: 3", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 3),
    ("Consistent 7", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 7),
]

# ---------- HELPERS ----------
def now_ts(): return time.time()

def clean_text_for_ai(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
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

# ---------- App panels ----------
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
                st.session_state["page"] = "Mindful Breathing"
                st.rerun()
        with c2:
            if st.button("Talk to AI"):
                st.session_state["page"] = "AI Doc Chat"
                st.rerun()
        with c3:
            if st.button("Journal"):
                st.session_state["page"] = "Mindful Journaling"
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
        mood = st.slider("How do you feel right now?", 1, 11, 6)
        st.markdown(f"**You chose:** {MOOD_EMOJI_MAP[mood]} · {mood}/11")
        note = st.text_input("Optional: Add a short note about why you feel this way")
        if st.button("Log Mood"):
            entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
            st.session_state["mood_history"].append(entry)

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
    st.markdown("Record a short message — the app will transcribe and reply.")

    if webrtc_streamer and AudioProcessorBase: # CHECK ADDED HERE
        class AudioProcessor(AudioProcessorBase):
            def __init__(self):
                self.audio_container = []

            def recv(self, frame):
                self.audio_container.append(frame.to_ndarray())
                return frame
        
        ctx = webrtc_streamer(
            key="audio-recorder",
            mode="sendonly",
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
        )

        if st.button("Stop Recording"):
            if ctx and ctx.state.playing:
                ctx.state.playing = False
                st.success("Recording stopped.")
                if ctx.audio_processor:
                    audio_data = ctx.audio_processor.audio_container
                    if audio_data:
                        # This is a mock transcription as STT API is not available
                        trans = "This is a placeholder transcription of the audio. You would replace this with a real STT when available."
                        st.session_state["transcription_text"] = trans
                        st.session_state["call_history"].append({"speaker":"User","text":trans,"timestamp":now_ts()})
                        
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
    else:
        st.warning("Audio recording is not available. Please ensure 'streamlit-webrtc' is installed.")
        st.info("You can still use the regular AI Doc Chat to type your message.")
        
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
            st.balloons()
            st.rerun()
        st.session_state.breath_running = False
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
def wellness_check_in_panel():
    st.header("Wellness Check-in")
    st.markdown("This check-in is based on the **Patient Health Questionnaire (PHQ-9)**, a widely used tool for depression screening. It is a tool for self-reflection and **not a professional diagnosis**.")

    st.markdown("### Over the past two weeks, how often have you been bothered by the following?")
    
    phq_questions = [
        "1. Little interest or pleasure in doing things?",
        "2. Feeling down, depressed, or hopeless?",
        "3. Trouble falling or staying asleep, or sleeping too much?",
        "4. Feeling tired or having little energy?",
        "5. Poor appetite or overeating?",
        "6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
        "7. Trouble concentrating on things, such as reading the newspaper or watching television?",
        "8. Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
        "9. Thoughts that you would be better off dead, or of hurting yourself in some way?"
    ]
    
    scores = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    
    answers = {}
    
    with st.form("phq9_form"):
        for i, q in enumerate(phq_questions):
            response = st.radio(q, list(scores.keys()), key=f"phq9_q{i}")
            answers[q] = response
        
        submitted = st.form_submit_button("Get My Score")

    if submitted:
        total_score = sum(scores[answers[q]] for q in phq_questions)
        
        interpretation = ""
        if total_score >= 20:
            interpretation = "Severe: A high score suggests severe symptoms. It is strongly recommended that you speak to a professional."
        elif total_score >= 15:
            interpretation = "Moderately Severe: A high score suggests moderately severe symptoms. You should consider speaking with a professional."
        elif total_score >= 10:
            interpretation = "Moderate: A moderate score suggests a need for support. Talking to a professional can be very helpful."
        elif total_score >= 5:
            interpretation = "Mild: A mild score suggests some symptoms, and the app's features may be a great help for self-care."
        else:
            interpretation = "Minimal to None: Your score suggests you're doing well. Keep up the self-care practices!"
        
        st.session_state.phq9_score = total_score
        st.session_state.phq9_interpretation = interpretation

        if "Wellness Check-in Completed" not in st.session_state["streaks"]["badges"]:
            st.session_state["streaks"]["badges"].append("Wellness Check-in Completed")

    if st.session_state.phq9_score is not None:
        st.subheader("Your Score")
        st.markdown(f"**{st.session_state.phq9_score}** out of 27")
        
        if "Severe" in st.session_state.phq9_interpretation:
            st.error(st.session_state.phq9_interpretation)
        elif "Moderately Severe" in st.session_state.phq9_interpretation:
            st.warning(st.session_state.phq9_interpretation)
        elif "Moderate" in st.session_state.phq9_interpretation:
            st.info(st.session_state.phq9_interpretation)
        else:
            st.success(st.session_state.phq9_interpretation)
            
        st.markdown("---")
        st.info("Remember, this is a screening tool, not a diagnosis. Please reach out to a professional for a full evaluation.")
        
        if st.session_state.phq9_score >= 10:
            st.subheader("Need Immediate Support?")
            st.warning("**If you are in crisis, please call or text the National Crisis and Suicide Lifeline: 988**")
            st.markdown("If you need to connect with a professional, we can help you find one.")
            st.markdown("### [Find a Professional near me](https://www.google.com/search?q=find+a+mental+health+professional)")
        
        if st.button("Take the test again"):
            st.session_state.phq9_score = None
            st.session_state.phq9_interpretation = None
            st.rerun()

def emotional_journey_panel():
    st.header("My Emotional Journey")
    all_text = get_all_user_text()
    if not all_text:
        st.info("Interact with the app more to build an emotional journey.")
        return
    st.subheader("AI-generated narrative (empathetic)")
    prompt = f"""
Write a short, supportive, and strength-focused 3-paragraph narrative about a person's recent emotional journey.
Use empathetic tone and offer gentle encouragement. Data:
{all_text[:4000]}
"""
    if ai_available:
        try:
            story = model.generate_content(prompt).text
            st.markdown(story)
            return
        except Exception:
            st.warning("AI generation failed. This might be a temporary issue with the service. A fallback narrative is being displayed.")
    
    fallback_story = "You’ve been carrying a lot — and showing up to this app is a small brave step. Over time, small acts of care add up. Keep logging your moments and celebrate tiny wins."
    st.markdown(fallback_story)

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
            entries.append({
                "date": datetime.fromtimestamp(ch.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S"),
                "text": ch.get("text"),
                "sentiment": sentiment_compound(ch.get("text", "")),
            })
    df = pd.DataFrame(entries)

    pos = len(df[df["sentiment"] >= 0.05]) if not df.empty else 0
    neg = len(df[df["sentiment"] <= -0.05]) if not df.empty else 0
    neu = len(df) - pos - neg if not df.empty else 0

    st.subheader("Sentiment Breakdown")
    st.write(f"- Positive entries: {pos}")
    st.write(f"- Neutral entries: {neu}")
    st.write(f"- Negative entries: {neg}")

    if ai_available:
        try:
            summary = model.generate_content(
                f"Summarize this person’s emotional trends in a supportive way:\n\n{all_text[:4000]}"
            ).text
        except Exception:
            summary = "Summary unavailable due to AI error."
    else:
        summary = "Based on your recent entries, you’re showing resilience and self-awareness. Keep going!"

    st.subheader("AI Summary")
    st.markdown(summary)

    if st.button("Export as PDF"):
        if pdf_canvas:
            buffer = io.BytesIO()
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            text_obj = c.beginText(40, 750)
            text_obj.textLine("Personalized Wellness Report")
            text_obj.textLine("")
            text_obj.textLines(summary)
            c.drawText(text_obj)
            c.showPage()
            c.save()

def crisis_support_panel():
    st.header("Crisis Support")
    st.markdown("### National Crisis and Suicide Lifeline")
    st.markdown("If you or someone you know is in emotional distress or suicidal crisis, you can call or text the **988 Suicide & Crisis Lifeline**.")
    st.markdown("### [Call or Text 988](tel:988)")
    st.markdown("### Other Resources")
    st.markdown("- **Crisis Text Line:** Text HOME to 741741 from anywhere in the US, anytime, about any type of crisis.")
    st.markdown("- **The Trevor Project:** Call 1-866-488-7386 or text START to 678-678. (For LGBTQ youth)")
    st.markdown("---")
    st.info("Remember, these services are free, confidential, and available 24/7.")


def main():
    st.sidebar.title("Navigation")
    sidebar_auth()
    
    pages = {
        "Home": homepage_panel,
        "Mood Tracker": mood_tracker_panel,
        "Wellness Check-in": wellness_check_in_panel,
        "AI Doc Chat": ai_doc_chat_panel,
        "Call Session": call_session_panel,
        "Mindful Breathing": mindful_breathing_panel,
        "Mindful Journaling": mindful_journaling_panel,
        "Journal & Analysis": journal_analysis_panel,
        "My Emotional Journey": emotional_journey_panel,
        "Personalized Report": personalized_report_panel,
        "Crisis Support": crisis_support_panel
    }
    
    page_names = list(pages.keys())
    
    # Initialize session state for the page if it doesn't exist
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    try:
        current_page_index = page_names.index(st.session_state.get("page"))
    except ValueError:
        current_page_index = 0
        st.session_state["page"] = "Home"

    page = st.sidebar.radio("Go to:", page_names, index=current_page_index)
    st.session_state["page"] = page
    
    func = pages.get(st.session_state.get("page"))
    if func:
        func()
    
    st.markdown("---")
    st.markdown("Built with care • Data stored locally unless you log in and save to your account.")

if __name__ == "__main__":
    main()