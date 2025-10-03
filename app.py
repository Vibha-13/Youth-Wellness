"""
AI Wellness Companion - Streamlit app (updated & optimized)
Features:
- Mood Tracker (badges, streaks)
- AI Chat (Gemini optional, lazy init)
- Journaling & Analysis (sentiment via VADER)
- Mindful Breathing (non-blocking-ish)
- PHQ-9 Wellness Check-in (complete)
- User Authentication (local + optional Supabase)
- Data persistence with Supabase (lazy)
- Export small PDF report (optional, lazy import reportlab)
"""

import streamlit as st
import os
import time
import random
import io
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Lightweight sentiment analyzer cached
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- CONSTANTS ----------
QUOTES = [
    "You are the only one who can limit your greatness. — Unknown",
    "I have chosen to be happy because it is good for my health. — Voltaire",
    "A sad soul can kill you quicker, far quicker than a germ. — John Steinbeck",
    "The groundwork for all happiness is health. — Leigh Hunt",
    "A calm mind brings inner strength and self-confidence. — Dalai Lama"
]

MOOD_EMOJI_MAP = {
    1: "😭 Agonizing", 2: "😩 Miserable", 3: "😞 Very Sad",
    4: "🙁 Sad", 5: "😐 Neutral/Okay", 6: "🙂 Content",
    7: "😊 Happy", 8: "😁 Very Happy", 9: "🤩 Excited",
    10: "🥳 Joyful", 11: "🌟 Fantastic"
}

BADGE_RULES = [
    ("First Log", lambda s: len(s["mood_history"]) >= 1),
    ("3-Day Streak", lambda s: s["streaks"].get("mood_log", 0) >= 3),
    ("Consistent Logger", lambda s: len(s["mood_history"]) >= 10),
    ("High Roller", lambda s: any(e.get("mood", 0) >= 10 for e in s["mood_history"])),
    ("Self-Aware", lambda s: len(s["mood_history"]) >= 5 and s["streaks"].get("mood_log", 0) >= 5)
]

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# ---------- CACHING & LAZY SETUP ----------
@st.cache_resource
def setup_analyzer():
    return SentimentIntensityAnalyzer()

# Lazy AI setup — defer heavy import until called
@st.cache_resource(show_spinner=False)
def setup_ai_model(api_key: str):
    """Lazy configure google.generativeai if available and key provided.
       Returns (model_obj or None, boolean ai_available)
    """
    if not api_key:
        return None, False
    try:
        # local import to avoid slowing module load
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")  # choose a small/fast model
        return model, True
    except Exception:
        return None, False

# Lazy supabase client setup (defer import)
@st.cache_resource(show_spinner=False)
def setup_supabase_client(url: str, key: str):
    if not url or not key:
        return None, False
    try:
        from supabase import create_client  # type: ignore
        client = create_client(url, key)
        return client, True
    except Exception:
        return None, False

# ---------- Session state defaults ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "call_history" not in st.session_state:
    st.session_state["call_history"] = []

if "daily_journal" not in st.session_state:
    st.session_state["daily_journal"] = []

if "mood_history" not in st.session_state:
    st.session_state["mood_history"] = []

if "streaks" not in st.session_state:
    st.session_state["streaks"] = {"mood_log": 0, "last_mood_date": None, "badges": []}

if "transcription_text" not in st.session_state:
    st.session_state["transcription_text"] = ""

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if "phq9_score" not in st.session_state:
    st.session_state["phq9_score"] = None

if "phq9_interpretation" not in st.session_state:
    st.session_state["phq9_interpretation"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [{"role": "assistant", "content": "Hello — I’m here to listen. What’s on your mind today?"}]

analyzer = setup_analyzer()

# ---------- Helper functions ----------
def now_ts():
    return time.time()

def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300):
    """Generate text via Gemini if available else fallback."""
    prompt_clean = clean_text_for_ai(prompt)
    # do not call heavy imports here; use cached setup function
    model, ai_available = st.session_state.get("_ai_model", (None, False))
    if ai_available and model:
        try:
            resp = model.generate_content(prompt_clean, max_output_tokens=max_tokens)
            text = getattr(resp, "text", None) or str(resp)
            return text
        except Exception:
            # fallback to canned replies
            pass
    canned = [
        "Thanks for sharing. I hear you — would you like to tell me more?",
        "That’s a lot to carry. I’m here. Could you describe one small thing that feels heavy right now?",
        "I’m listening. If you want, we can try a 1-minute breathing exercise together."
    ]
    return random.choice(canned)

def sentiment_compound(text: str) -> float:
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def get_all_user_text() -> str:
    parts = []
    parts += [e.get("text","") for e in st.session_state["daily_journal"] if e.get("text")]
    parts += [m.get("content","") for m in st.session_state["chat_messages"] if m.get("role") == "user" and m.get("content")]
    parts += [c.get("text","") for c in st.session_state["call_history"] if c.get("speaker") == "User" and c.get("text")]
    return " ".join(parts).strip()

def generate_wordcloud_figure_if_possible(text: str):
    if not text or not text.strip():
        return None
    try:
        # lazy import
        from wordcloud import WordCloud  # type: ignore
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception:
        return None

# ---------- Supabase helpers (lazy & guarded) ----------
def register_user_db(email: str):
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return None
    try:
        res = supabase_client.table("users").insert({"email": email}).execute()
        if getattr(res, "data", None):
            return res.data[0].get("id")
    except Exception:
        return None

def get_user_by_email_db(email: str):
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return []
    try:
        res = supabase_client.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def load_journal_db(user_id, supabase_client):
    if not supabase_client:
        return []
    try:
        res = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

# ---------- UI style ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; font-family: 'Poppins', sans-serif; }
    .main .block-container { padding: 2rem 3rem; }
    .card { background-color: #eaf4ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 18px; margin-bottom: 18px; border-left: 5px solid #4a90e2; transition: transform .12s; }
    .card:hover { transform: translateY(-4px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
    .stButton>button { color: #fff; background-color: #4a90e2; border-radius: 8px; padding: 8px 18px; font-weight:600; border: none; }
    .stButton>button:hover { background-color: #357bd9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Services setup (executed quickly) ----------
# Use secrets (or env) but do lazy configure with setup functions
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
_ai_model, _ai_available = setup_ai_model(GEMINI_API_KEY)
# store in session_state for safe_generate to access quickly
st.session_state["_ai_model"] = (_ai_model, _ai_available)
st.session_state["_ai_available"] = _ai_available

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
_supabase_client_obj, _db_connected = setup_supabase_client(SUPABASE_URL, SUPABASE_KEY)
st.session_state["_supabase_client_obj"] = _supabase_client_obj
st.session_state["_db_connected"] = _db_connected

st.sidebar.markdown(f"- AI: **{'Connected' if _ai_available else 'Local (fallback)'}**")
st.sidebar.markdown(f"- DB: **{'Connected' if _db_connected else 'Not connected'}**")

# ---------- Sidebar: Auth ----------
def sidebar_auth():
    st.sidebar.header("Account")
    if not st.session_state.get("logged_in"):
        email = st.sidebar.text_input("Your email", key="login_email")
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                if st.session_state.get("_db_connected"):
                    user_list = get_user_by_email_db(email)
                    if user_list:
                        user = user_list[0]
                if user:
                    st.session_state["user_id"] = user.get("id")
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    entries = load_journal_db(st.session_state["user_id"], st.session_state.get("_supabase_client_obj")) or []
                    st.session_state["daily_journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in entries]
                    st.sidebar.success("Logged in.")
                    st.experimental_rerun()
                else:
                    uid = None
                    if st.session_state.get("_db_connected"):
                        uid = register_user_db(email)
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.sidebar.success("Registered & logged in.")
                        st.experimental_rerun()
                    else:
                        st.session_state["logged_in"] = True
                        st.session_state["user_email"] = email
                        st.sidebar.info("Logged in locally (no DB).")
                        st.experimental_rerun()
            else:
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            # Preserve minimal state but clear user-specific entries
            st.session_state["logged_in"] = False
            st.session_state["user_id"] = None
            st.session_state["user_email"] = None
            st.sidebar.info("Logged out.")
            st.experimental_rerun()

# ---------- Panels ----------
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
                st.experimental_rerun()
        with c2:
            if st.button("Talk to AI"):
                st.session_state["page"] = "AI Chat"
                st.experimental_rerun()
        with c3:
            if st.button("Journal"):
                st.session_state["page"] = "Mindful Journaling"
                st.experimental_rerun()
    with col2:
        st.image("https://images.unsplash.com/photo-1549490349-f06b3e942007?q=80&w=2070&auto=format&fit=crop", caption="Take a moment for yourself")
    st.markdown("---")
    st.header("Features")
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown("#### Mood Tracker")
        st.markdown("Log quick mood ratings and unlock badges.")
    with f2:
        st.markdown("#### AI Chat")
        st.markdown("A compassionate AI to listen and suggest small exercises.")
    with f3:
        st.markdown("#### Journal & Insights")
        st.markdown("Track progress over time with charts and word clouds.")

def mood_tracker_panel():
    st.header("Daily Mood Tracker")
    col1, col2 = st.columns([3,1])
    with col1:
        mood = st.slider("How do you feel right now?", 1, 11, 6)
        st.markdown(f"**You chose:** {MOOD_EMOJI_MAP.get(mood, 'N/A')} · {mood}/11")
        note = st.text_input("Optional: Add a short note about why you feel this way")
        if st.button("Log Mood"):
            entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
            st.session_state["mood_history"].append(entry)

            last_date = st.session_state["streaks"].get("last_mood_date")
            today = datetime.now().date()
            last_dt = None
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
                except Exception:
                    last_dt = None

            if last_dt != today:
                yesterday = today - timedelta(days=1)
                if last_dt == yesterday:
                    st.session_state["streaks"]["mood_log"] = st.session_state["streaks"].get("mood_log", 0) + 1
                else:
                    st.session_state["streaks"]["mood_log"] = 1
                st.session_state["streaks"]["last_mood_date"] = today.strftime("%Y-%m-%d")

            st.success("Mood logged. Tiny step, big impact.")

            # Badge check
            for name, rule in BADGE_RULES:
                try:
                    state_subset = {"mood_history": st.session_state["mood_history"], "streaks": st.session_state["streaks"]}
                    if rule(state_subset):
                        if name not in st.session_state["streaks"]["badges"]:
                            st.session_state["streaks"]["badges"].append(name)
                except Exception:
                    continue
            st.experimental_rerun()

    with col2:
        st.subheader("Badges")
        if st.session_state["streaks"]["badges"]:
            for b in st.session_state["streaks"]["badges"]:
                # st.badge may not exist in all streamlit versions; fallback to markdown
                try:
                    st.badge(b, color="yellow")
                except Exception:
                    st.markdown(f"- **{b}**")
        else:
            st.markdown("_No badges yet — log a mood to get started!_")

        st.subheader("Streak")
        st.markdown(f"Consecutive days logging mood: **{st.session_state['streaks'].get('mood_log',0)}**")

    # Plot mood history if exists
    if st.session_state["mood_history"]:
        df = pd.DataFrame(st.session_state["mood_history"]).copy()
        df['date'] = pd.to_datetime(df['date'])
        fig = px.line(df, x='date', y='mood', title="Mood Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def ai_chat_panel():
    st.header("AI Chat")
    st.markdown("A compassionate AI buddy to listen.")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What's on your mind?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # ensure ai model is present in session
                model, ai_available = st.session_state.get("_ai_model", (None, False))
                if not ai_available:
                    # try to set up once more quickly
                    model, ai_available = setup_ai_model(GEMINI_API_KEY)
                    st.session_state["_ai_model"] = (model, ai_available)
                    st.session_state["_ai_available"] = ai_available

                ai_response = safe_generate(prompt)
                st.markdown(ai_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
        # do not force rerun here — keep chat responsive

def mindful_breathing_panel():
    st.header("Mindful Breathing")
    st.markdown("Follow the prompts: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Try 3 cycles.")
    # We'll implement a simple non-blocking-ish timer using session state timestamps
    if "breathing_state" not in st.session_state:
        st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}

    bs = st.session_state["breathing_state"]
    start_btn = st.button("Start Exercise", key="start_breathing_btn")
    if start_btn and not bs["running"]:
        bs["running"] = True
        bs["start_time"] = time.time()
        bs["cycles_done"] = 0
        st.session_state["breathing_state"] = bs
        st.experimental_rerun()

    if bs["running"]:
        PHASES = [("Inhale", 4.0, "#4a90e2"), ("Hold", 4.0, "#357bd9"), ("Exhale", 6.0, "#f39c12")]
        total_cycle_time = sum(p[1] for p in PHASES)
        elapsed = time.time() - (bs["start_time"] or time.time())
        cycle_number = int(elapsed // total_cycle_time) + 1
        time_in_cycle = elapsed % total_cycle_time

        if cycle_number > 3:
            bs["running"] = False
            bs["cycles_done"] = 3
            st.session_state["breathing_state"] = bs
            st.success("Exercise complete! You did a great job.")
            return

        st.info(f"Cycle {cycle_number} of 3")
        phase_start = 0.0
        for phase, duration, color in PHASES:
            if time_in_cycle < phase_start + duration:
                time_in_phase = time_in_cycle - phase_start
                progress = min(max(time_in_phase / duration, 0.0), 1.0)
                st.markdown(f"<h2 style='text-align:center;color:{color};'>{phase}</h2>", unsafe_allow_html=True)
                st.progress(progress)
                break
            phase_start += duration

        # refresh frequently but not blocking
        time.sleep(0.15)
        st.experimental_rerun()

    if not bs["running"] and bs["cycles_done"] < 3:
        if st.button("Reset", key="reset_breathing_btn"):
            st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}
            st.experimental_rerun()

def mindful_journaling_panel():
    st.header("Mindful Journaling")
    st.markdown("Write freely — your words are private here unless you save to your account.")
    journal_text = st.text_area("Today's reflection", height=220, key="journal_text")
    if st.button("Save Entry"):
        if journal_text.strip():
            sent = sentiment_compound(journal_text)
            # Persist to DB if available, else save local
            if st.session_state.get("logged_in") and st.session_state.get("_db_connected") and st.session_state.get("user_id"):
                ok = save_journal_db(st.session_state.get("user_id"), journal_text, sent)
                if ok:
                    st.success("Saved to your account.")
                else:
                    st.warning("Could not save to DB. Saved locally instead.")
                    st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
            else:
                st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
                st.success("Saved locally.")
            st.experimental_rerun()
        else:
            st.warning("Write something you want to save.")

def journal_analysis_panel():
    st.header("Journal & Analysis")
    all_text = get_all_user_text()
    if not all_text:
        st.info("No journal or chat text yet — start journaling or talking to get insights.")
        return

    # Build entries list
    entries = []
    for e in st.session_state["daily_journal"]:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e.get("sentiment", 0)})
    # user chat entries
    chat_entries = [{"date": datetime.now(), "compound": sentiment_compound(msg["content"])} for msg in st.session_state.chat_messages if msg["role"] == "user"]
    entries.extend(chat_entries)

    if entries:
        df = pd.DataFrame(entries).sort_values("date")
        df["sentiment_label"] = df["compound"].apply(lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
        fig = px.line(df, x="date", y="compound", color="sentiment_label", markers=True,
                      title="Sentiment Over Time",
                      color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"})
        st.plotly_chart(fig, use_container_width=True)

    wc_fig = generate_wordcloud_figure_if_possible(all_text)
    if wc_fig:
        st.subheader("Word Cloud")
        st.pyplot(wc_fig, clear_figure=True)

def wellness_check_in_panel():
    st.header("Wellness Check-in (PHQ-9)")
    st.markdown("This check-in is a screening tool and not a diagnosis. If you're worried about your mental health, consider speaking to a professional.")

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

    with st.form("phq9_form"):
        answers = {}
        for i, q in enumerate(phq_questions):
            response = st.radio(q, list(scores.keys()), key=f"phq9_q{i}")
            answers[q] = response
        submitted = st.form_submit_button("Get My Score")

    if submitted:
        total_score = sum(scores[answers[q]] for q in phq_questions)
        interpretation = ""
        if total_score >= 20:
            interpretation = "Severe: A high score suggests severe symptoms. It is strongly recommended you seek professional help."
        elif total_score >= 15:
            interpretation = "Moderately Severe: Consider making an appointment with a mental health professional."
        elif total_score >= 10:
            interpretation = "Moderate: You may benefit from talking to a professional or increased self-care."
        elif total_score >= 5:
            interpretation = "Mild: Some symptoms are present; keep monitoring and using self-care practices."
        else:
            interpretation = "Minimal to None: Your score suggests few or no symptoms at present."

        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation

        # Badge awarding
        if "Wellness Check-in Completed" not in st.session_state["streaks"]["badges"]:
            st.session_state["streaks"]["badges"].append("Wellness Check-in Completed")

    # Display results if present
    if st.session_state.get("phq9_score") is not None:
        st.subheader("Your Score")
        st.markdown(f"**{st.session_state['phq9_score']}** out of 27")
        if "Severe" in st.session_state["phq9_interpretation"]:
            st.error(st.session_state["phq9_interpretation"])
        elif "Moderately Severe" in st.session_state["phq9_interpretation"]:
            st.warning(st.session_state["phq9_interpretation"])
        elif "Moderate" in st.session_state["phq9_interpretation"]:
            st.info(st.session_state["phq9_interpretation"])
        else:
            st.success(st.session_state["phq9_interpretation"])

        st.markdown("---")
        st.info("If you are in crisis or feel you might harm yourself, please call local emergency services immediately.")
        # Provide USA helpline as a convenience and generic suggestion
        st.markdown("If you are in the United States, call or text **988** for immediate support. If elsewhere, check local resources or a trusted person.")

        if st.button("Reset PHQ-9"):
            st.session_state["phq9_score"] = None
            st.session_state["phq9_interpretation"] = None
            st.experimental_rerun()

        # Offer PDF export if reportlab installed
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas as pdf_canvas  # type: ignore
            can_export = True
        except Exception:
            can_export = False

        if can_export and st.button("Export PHQ-9 as PDF"):
            buffer = io.BytesIO()
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            y = 750
            c.drawString(40, y, "PHQ-9 Wellness Check Report")
            y -= 30
            c.setFont("Helvetica", 12)
            c.drawString(40, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 20
            c.drawString(40, y, f"Score: {st.session_state['phq9_score']} / 27")
            y -= 20
            c.drawString(40, y, f"Interpretation: {st.session_state['phq9_interpretation']}")
            y -= 30
            c.drawString(40, y, "Answers:")
            y -= 20
            for i, q in enumerate(phq_questions):
                ans = st.session_state.get(f"phq9_q{i}", "Not answered")
                # safety: truncate long strings
                c.drawString(50, y, f"{q} — {ans[:80]}")
                y -= 14
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750
            c.save()
            st.download_button(label="Download PHQ-9 Report", data=buffer.getvalue(), file_name="phq9_report.pdf", mime="application/pdf")

def emotional_journey_panel():
    st.header("My Emotional Journey")
    all_text = get_all_user_text()
    if not all_text:
        st.info("Interact with the app more to build an emotional journey.")
        return

    st.subheader("AI-generated narrative (empathetic)")
    prompt = f"""
Write a short, supportive, and strength-focused 3-paragraph narrative about a person's recent emotional journey.
Use an empathetic tone and offer gentle encouragement. Data:
{all_text[:4000]}
"""
    # attempt to use AI if available
    model, available = st.session_state.get("_ai_model", (None, False))
    if available:
        try:
            story = safe_generate(prompt, max_tokens=400)
            st.markdown(story)
            return
        except Exception:
            st.warning("AI generation failed — showing fallback narrative.")

    fallback = "You’ve been carrying a lot — and showing up to this app is a small brave step. Over time, small acts of care add up. Keep logging your moments and celebrate tiny wins."
    st.markdown(fallback)

def personalized_report_panel():
    st.header("Personalized Report")
    all_text = get_all_user_text()
    if not all_text:
        st.info("No data yet. Start journaling or chatting to generate a report.")
        return

    entries = []
    for e in st.session_state["daily_journal"]:
        entries.append({"date": e.get("date"), "text": e.get("text"), "sentiment": e.get("sentiment", 0)})
    for msg in st.session_state["chat_messages"]:
        if msg.get("role") == "user":
            entries.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": msg.get("content"), "sentiment": sentiment_compound(msg.get("content",""))})

    df = pd.DataFrame(entries)
    if not df.empty:
        pos = len(df[df["sentiment"] >= 0.05])
        neg = len(df[df["sentiment"] <= -0.05])
        neu = len(df) - pos - neg
    else:
        pos = neg = neu = 0

    st.subheader("Sentiment Breakdown")
    st.write(f"- Positive entries: {pos}")
    st.write(f"- Neutral entries: {neu}")
    st.write(f"- Negative entries: {neg}")

    summary = "Summary unavailable."
    if st.session_state.get("_ai_available"):
        try:
            summary = safe_generate(f"Summarize this person’s emotional trends in a supportive way:\n\n{all_text[:4000]}", max_tokens=220)
        except Exception:
            summary = "Based on recent entries, you show resilience. Keep small self-care habits."
    else:
        summary = "Based on your recent entries, you’re showing resilience and self-awareness. Keep going!"

    st.subheader("AI Summary")
    st.markdown(summary)

    if st.button("Export as PDF (brief)"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas as pdf_canvas  # type: ignore
            buffer = io.BytesIO()
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            y = 750
            c.drawString(40, y, "Personalized Wellness Report")
            y -= 30
            c.setFont("Helvetica", 12)
            for line in summary.split("\n"):
                c.drawString(40, y, line[:110])
                y -= 14
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750
            c.save()
            st.download_button("Download Report PDF", buffer.getvalue(), file_name="wellness_report.pdf", mime="application/pdf")
        except Exception:
            st.error("PDF generation not available (reportlab missing).")

def crisis_support_panel():
    st.header("Crisis Support")
    st.markdown("If you or someone you know is in immediate danger, contact local emergency services.")
    st.markdown("**If you are in the United States and are in crisis, call or text 988.**")
    st.markdown("- Crisis Text Line: Text HOME to 741741 (US)")
    st.markdown("- The Trevor Project (LGBTQ+ youth): 1-866-488-7386 or text START to 678678")
    st.markdown("---")
    st.info("These services are free, confidential, and available 24/7. If you are outside the US, please consult local resources.")

def progress_dashboard_panel():
    st.header("Progress Dashboard")
    st.markdown("A quick glance at your recent mood logs and PHQ-9 (if completed).")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["mood_history"]:
            df = pd.DataFrame(st.session_state["mood_history"])
            df['date'] = pd.to_datetime(df['date'])
            recent = df.sort_values("date").tail(7)
            fig = px.bar(recent, x='date', y='mood', title="Last 7 Mood Logs")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mood logs yet.")

    with col2:
        if st.session_state.get("phq9_score") is not None:
            st.metric("Last PHQ-9 score", st.session_state["phq9_score"])
            st.markdown(f"**Interpretation:** {st.session_state['phq9_interpretation']}")
        else:
            st.info("Complete a PHQ-9 check-in to see results here.")

# ---------- Main app ----------
def main():
    st.sidebar.title("Navigation")
    sidebar_auth()

    pages = {
        "Home": homepage_panel,
        "Mood Tracker": mood_tracker_panel,
        "Wellness Check-in": wellness_check_in_panel,
        "AI Chat": ai_chat_panel,
        "Mindful Breathing": mindful_breathing_panel,
        "Mindful Journaling": mindful_journaling_panel,
        "Journal & Analysis": journal_analysis_panel,
        "My Emotional Journey": emotional_journey_panel,
        "Personalized Report": personalized_report_panel,
        "Crisis Support": crisis_support_panel,
        "Progress Dashboard": progress_dashboard_panel
    }

    page_names = list(pages.keys())
    try:
        current_page_index = page_names.index(st.session_state.get("page"))
    except ValueError:
        current_page_index = 0
        st.session_state["page"] = "Home"

    page = st.sidebar.radio("Go to:", page_names, index=current_page_index)
    st.session_state["page"] = page

    func = pages.get(page)
    if func:
        func()

    st.markdown("---")
    st.markdown("Built with care • Data stored locally unless you log in and save to your account.")

if __name__ == "__main__":
    main()
