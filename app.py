# app.py
import streamlit as st
import os
import time
import random
import io
import math
from datetime import datetime
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# AI and NLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Audio
try:
    import sounddevice as sd
    import wavio
except Exception:
    sd = None
    wavio = None

# TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Supabase client (optional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# ---------- STYLES (pastel wellness) ----------
st.markdown(
    """
    <style>
    /* Page background */
    [data-testid="stAppViewContainer"]{
        background: linear-gradient(180deg,#fff8fb 0%, #f2f7ff 100%);
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }
    /* Sidebar */
    [data-testid="stSidebar"]{
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 16px;
    }
    /* Card */
    .card {
        background: white;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(99,102,241,0.08);
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#8ec5ff,#e0c3fc);
        color: #0b1220;
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 600;
    }
    /* Small helper */
    .muted { color: #6b7280; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SECRETS & EXTERNAL SERVICES ----------
# Gemini (Google) API
GEMINI_API_KEY = None
if st.secrets and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
elif os.getenv("GEMINI_API_KEY"):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        ai_available = True
    except Exception as e:
        st.sidebar.warning("AI API connection failed. Running in local fallback mode.")
        ai_available = False
else:
    ai_available = False

# Supabase (optional)
SUPABASE_URL = None
SUPABASE_KEY = None
supabase = None
if st.secrets and "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
elif os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        db_connected = True
    except Exception:
        db_connected = False
else:
    db_connected = False

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history (list of {'role','content','ts'})
if "call_history" not in st.session_state:
    st.session_state.call_history = []
if "daily_journal" not in st.session_state:
    st.session_state.daily_journal = []  # local fallback
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "streaks" not in st.session_state:
    st.session_state.streaks = {"mood_log": 0, "last_mood_date": None, "badges": []}
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

analyzer = SentimentIntensityAnalyzer()

# ---------- CONTENT & UTILS ----------
QUOTES = [
    "You are stronger than you think. 💪",
    "Even small steps count. 🌱",
    "Breathe. You are doing your best. 🌬️",
    "This moment will pass. You're doing important work by being here. 💛",
    "Progress, not perfection. Tiny steps add up."
]

MOOD_EMOJI_MAP = {
    1: "😭", 2: "😢", 3: "😔", 4: "😕", 5: "😐",
    6: "🙂", 7: "😊", 8: "😄", 9: "🤩", 10: "🥳"
}

BADGE_RULES = [
    ("Getting Started", lambda s: len(s["mood_history"]) >= 1),
    ("Weekly Streak: 3", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 3),
    ("Consistent 7", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 7),
]

def now_ts():
    return time.time()

def safe_generate(prompt: str, max_tokens: int = 300):
    """Generate text using Gemini if available, otherwise fallback canned reply."""
    if ai_available:
        try:
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            st.warning("AI generation failed — using fallback.")
    # fallback: short empathetic rewrite
    canned = [
        "Thanks for sharing. I hear you — would you like to tell me more about what’s been going on?",
        "That’s a lot to carry. I’m here with you. Could you describe one thing that feels heavy right now?",
        "I’m listening. If you want, we can try a 1-minute breathing exercise together."
    ]
    return random.choice(canned)

def sentiment_compound(text):
    return analyzer.polarity_scores(text)["compound"]

def generate_wordcloud_figure(text):
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def record_audio(duration=5, fs=44100):
    if sd is None or wavio is None:
        st.warning("Recording not available on this environment.")
        return None
    st.info("Recording... speak now.")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True, dtype='int16')
    st.success("Recording complete.")
    memfile = io.BytesIO()
    wavio.write(memfile, audio_data, fs, sampwidth=2)
    memfile.seek(0)
    return memfile

def tts_play(text):
    # Try pyttsx3 (local) TTS. If not available, skip.
    if pyttsx3 is None:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning("Local TTS failed.")

# ---------- DATABASE HELPERS (supabase) ----------
def register_user_db(email):
    if not db_connected:
        return None
    try:
        res = supabase.table("users").insert({"email": email}).execute()
        if res.data:
            return res.data[0]["id"]
    except Exception as e:
        st.warning("Supabase register failed.")
    return None

def get_user_by_email_db(email):
    if not db_connected:
        return None
    try:
        res = supabase.table("users").select("*").eq("email", email).execute()
        return res.data
    except Exception:
        return None

def save_journal_db(user_id, text, sentiment):
    if not db_connected:
        return False
    try:
        supabase.table("journal_entries").insert({
            "user_id": user_id,
            "entry_text": text,
            "sentiment_score": float(sentiment)
        }).execute()
        return True
    except Exception:
        return False

def load_journal_db(user_id):
    if not db_connected:
        return []
    try:
        res = supabase.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

# ---------- UI COMPONENTS ----------
def sidebar_auth():
    st.sidebar.header("Account")
    if not st.session_state.logged_in:
        email = st.sidebar.text_input("Your email", key="login_email")
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                if db_connected:
                    user = get_user_by_email_db(email)
                if user:
                    st.session_state.user_id = user[0]["id"]
                    st.session_state.user_email = email
                    st.session_state.logged_in = True
                    # load journals
                    entries = load_journal_db(st.session_state.user_id)
                    st.session_state.daily_journal = [
                        {"date": e["created_at"], "text": e["entry_text"], "sentiment": e["sentiment_score"]}
                        for e in entries
                    ]
                    st.sidebar.success("Logged in.")
                else:
                    uid = None
                    if db_connected:
                        uid = register_user_db(email)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.user_email = email
                        st.session_state.logged_in = True
                        st.sidebar.success("Registered & logged in.")
                    else:
                        # fallback local login
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.sidebar.info("Logged in locally (no DB).")
            else:
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.user_email}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.sidebar.info("Logged out.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    st.sidebar.markdown(f"- AI: {'Connected' if ai_available else 'Local fallback'}")
    st.sidebar.markdown(f"- DB: {'Connected' if db_connected else 'Not connected'}")

def header_home():
    st.title("Tools for Your Wellbeing")
    st.markdown("Discover resources, tools and micro-practices to support your mental health journey.")
    st.markdown("---")

def homepage_panel():
    header_home()
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Daily Inspiration")
        st.markdown(f"**{random.choice(QUOTES)}**")
        st.markdown("**Quick actions**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Start Breathing"):
                st.session_state.page = "breathing"
                st.experimental_rerun()
        with c2:
            if st.button("Talk to AI"):
                st.session_state.page = "chat"
                st.experimental_rerun()
        with c3:
            if st.button("Journal"):
                st.session_state.page = "journaling"
                st.experimental_rerun()

        st.markdown("### Mood Snapshot")
        if st.session_state.mood_history:
            last = st.session_state.mood_history[-1]
            st.markdown(f"**Last mood:** {MOOD_EMOJI_MAP.get(last['mood'], '')}  ·  {last['mood']}/10  ·  {last['date']}")
        else:
            st.info("Log your mood — it's quick and helps the AI personalize suggestions.")

    with col2:
        st.image("https://images.unsplash.com/photo-1505751172876-fa1923c5c528?q=80&w=1400&auto=format&fit=crop")

    st.markdown("---")
    st.markdown("### Features")
    f1, f2, f3 = st.columns(3)
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
        st.markdown(f"**You chose:** {MOOD_EMOJI_MAP[mood]}  ·  {mood}/10")
        note = st.text_input("Optional: Add a short note about why you feel this way")
        if st.button("Log Mood"):
            entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
            st.session_state.mood_history.append(entry)

            # update streaks (simple daily streak)
            last_date = st.session_state.streaks.get("last_mood_date")
            today = datetime.now().date()
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
                except Exception:
                    last_dt = None
            else:
                last_dt = None

            if last_dt == today:
                # already logged today -> don't change
                pass
            else:
                if last_dt == (today - pd.Timedelta(days=1).date()):
                    st.session_state.streaks["mood_log"] = st.session_state.streaks.get("mood_log", 0) + 1
                else:
                    st.session_state.streaks["mood_log"] = 1
                st.session_state.streaks["last_mood_date"] = today.strftime("%Y-%m-%d")
            st.success("Mood logged. Tiny step, big impact.")
            # unlock badges
            for name, rule in BADGE_RULES:
                if rule({"mood_history": st.session_state.mood_history, "streaks": st.session_state.streaks}):
                    if name not in st.session_state.streaks["badges"]:
                        st.session_state.streaks["badges"].append(name)
            st.experimental_rerun()

    with col2:
        st.markdown("### Badges")
        for b in st.session_state.streaks["badges"]:
            st.markdown(f"- 🏅 {b}")
        st.markdown("### Streak")
        st.markdown(f"Consecutive days logging mood: **{st.session_state.streaks.get('mood_log',0)}**")

    # timeline
    if st.session_state.mood_history:
        df = pd.DataFrame(st.session_state.mood_history)
        df['date'] = pd.to_datetime(df['date'])
        df_plot = df.copy()
        fig = px.line(df_plot, x='date', y='mood', title="Mood Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def ai_doc_chat_panel():
    st.header("AI Doc Chat")
    st.markdown("A compassionate AI buddy to listen. Your conversations remain private in this app (unless you opt-in to save).")
    # render conversation
    for msg in st.session_state.messages:
        role = msg.get("role","user")
        ts = msg.get("ts","")
        if role == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")
    user_input = st.text_input("What's on your mind?", key="chat_input")
    if st.button("Send", key="send_chat"):
        if user_input.strip():
            st.session_state.messages.append({"role":"user","content":user_input,"ts":now_ts()})
            # AI response (context aware)
            prompt_context = "\n\n".join([m["content"] for m in st.session_state.messages[-6:]])
            ai_resp = safe_generate(prompt_context)
            st.session_state.messages.append({"role":"assistant","content":ai_resp,"ts":now_ts()})
            st.experimental_rerun()

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
                # transcription placeholder
                trans = " ".join([
                    "This is a short transcription placeholder.",
                    "You can replace this with a real STT when available."
                ])
                st.session_state.transcription_text = trans
                st.session_state.call_history.append({"speaker":"User","text":trans,"timestamp":now_ts()})
                st.experimental_rerun()

    if st.session_state.transcription_text:
        st.markdown("**You said:**")
        st.info(st.session_state.transcription_text)
        if st.button("Get AI Reply"):
            st.session_state.messages.append({"role":"user","content":st.session_state.transcription_text,"ts":now_ts()})
            ai_resp = safe_generate(st.session_state.transcription_text)
            st.session_state.call_history.append({"speaker":"AI","text":ai_resp,"timestamp":now_ts()})
            st.markdown(f"**AI:** {ai_resp}")
            # TTS
            try:
                tts_play(ai_resp)
            except Exception:
                pass
            st.session_state.transcription_text = ""

    if st.session_state.call_history:
        st.markdown("---")
        st.subheader("Call History")
        for e in st.session_state.call_history[-10:]:
            who = e['speaker']
            txt = e['text']
            st.markdown(f"**{who}:** {txt}")

def mindful_breathing_panel():
    st.header("Mindful Breathing — 4-4-6 (Short)")
    st.markdown("Follow the prompts: Inhale (4s) — Hold (4s) — Exhale (6s). Try 3 cycles.")
    if "breath_running" not in st.session_state:
        st.session_state.breath_running = False
        st.session_state.breath_phase = ""
        st.session_state.breath_cycle = 0

    if st.button("Start Exercise"):
        st.session_state.breath_running = True
        st.session_state.breath_cycle = 0

    if st.button("Reset"):
        st.session_state.breath_running = False
        st.session_state.breath_phase = ""
        st.session_state.breath_cycle = 0

    if st.session_state.breath_running:
        cycles = 3
        pattern = [("Inhale", 4), ("Hold", 4), ("Exhale", 6)]
        for c in range(st.session_state.breath_cycle, cycles):
            for phase, sec in pattern:
                st.session_state.breath_phase = phase
                st.markdown(f"**{phase}** — {sec} seconds")
                placeholder = st.empty()
                for t in range(sec, 0, -1):
                    placeholder.markdown(f"<h2 style='color:#374151'>{t}</h2>", unsafe_allow_html=True)
                    time.sleep(1)
                placeholder.empty()
            st.session_state.breath_cycle = c+1
            st.experimental_rerun()
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
            if st.session_state.logged_in and db_connected and st.session_state.user_id:
                ok = save_journal_db(st.session_state.user_id, journal_text, sent)
                if ok:
                    st.success("Saved to your account.")
                else:
                    st.warning("Could not save to DB. Saved locally instead.")
                    st.session_state.daily_journal.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
            else:
                st.session_state.daily_journal.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
                st.success("Saved locally.")
            st.experimental_rerun()
        else:
            st.warning("Write something you want to save.")

def journal_analysis_panel():
    st.header("Journal & Analysis")
    all_text = " ".join([e["text"] for e in st.session_state.daily_journal]) + " " + " ".join([e["text"] for e in st.session_state.call_history if e["speaker"]=="User"])
    if not all_text.strip():
        st.info("No journal or call text yet — start journaling or talking to get insights.")
        return
    # sentiment timeline
    entries = []
    for e in st.session_state.daily_journal:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e["sentiment"]})
    for ch in st.session_state.call_history:
        if ch["speaker"]=="User":
            entries.append({"date": pd.to_datetime(datetime.fromtimestamp(ch["timestamp"])), "compound": sentiment_compound(ch["text"])})
    if entries:
        df = pd.DataFrame(entries).sort_values("date")
        df["sentiment_label"] = df["compound"].apply(lambda x: "Positive" if x>=0.05 else ("Negative" if x<=-0.05 else "Neutral"))
        fig = px.line(df, x="date", y="compound", color="sentiment_label", markers=True, title="Sentiment Over Time", color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"})
        st.plotly_chart(fig, use_container_width=True)
    # wordcloud
    wc_fig = generate_wordcloud_figure(all_text)
    if wc_fig:
        st.subheader("Word Cloud")
        st.pyplot(wc_fig)

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
        st.markdown("**Suggestion:**")
        st.info(suggestion)
        # small reward
        if "Mood Booster" not in st.session_state.streaks["badges"]:
            st.session_state.streaks["badges"].append("Mood Booster")
        st.experimental_rerun()

def emotional_journey_panel():
    st.header("Your Emotional Journey")
    all_text = " ".join([e["text"] for e in st.session_state.daily_journal]) + " " + " ".join([e["text"] for e in st.session_state.call_history if e["speaker"]=="User"])
    if not all_text.strip():
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
            st.warning("AI generation failed; showing fallback.")
    # fallback story
    fallback_story = "You’ve been carrying a lot — and showing up to this app is a small brave step. Over time, small acts of care add up. Keep logging your moments and celebrate tiny wins."
    st.markdown(fallback_story)

def personalized_report_panel():
    st.header("Personalized Report")
    all_text = " ".join([e["text"] for e in st.session_state.daily_journal]) + " " + " ".join([e["text"] for e in st.session_state.call_history if e["speaker"]=="User"])
    if not all_text.strip():
        st.info("No data yet. Start journaling or chatting to generate a report.")
        return
    # compute summary
    entries = []
    for e in st.session_state.daily_journal:
        entries.append(e)
    for ch in st.session_state.call_history:
        if ch["speaker"]=="User":
            entries.append({"date": datetime.fromtimestamp(ch["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"), "text": ch["text"], "sentiment": sentiment_compound(ch["text"])})
    df = pd.DataFrame(entries)
    pos = len(df[df["sentiment"]>=0.05])
    neg = len(df[df["sentiment"]<=-0.05])
    neut = len(df) - pos - neg
    st.markdown(f"**Entries analyzed:** {len(df)}")
    st.markdown(f"- Positive: {pos}")
    st.markdown(f"- Neutral: {neut}")
    st.markdown(f"- Negative: {neg}")
    # AI insight
    insight_prompt = f"Summarize the main emotional themes in these notes and give 3 gentle suggestions: {all_text[:4000]}"
    insight = safe_generate(insight_prompt)
    st.subheader("AI Insight")
    st.write(insight)
    # download
    report_text = f"Summary generated on {datetime.now().strftime('%Y-%m-%d')}\nEntries: {len(df)}\nPositive:{pos}\nNeutral:{neut}\nNegative:{neg}\n\nAI Insight:\n{insight}\n\nRaw text:\n{all_text}"
    st.download_button("Download Report", data=report_text, file_name="wellness_report.txt", mime="text/plain")

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
    # run page
    func = pages.get(page)
    if func:
        func()
    # small footer
    st.markdown("---")
    st.markdown("Built with care • Data stored locally unless you log in and save to your account.")

if __name__ == "__main__":
    main()
