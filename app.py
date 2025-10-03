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

# Supabase (optional)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Optional PDF generator
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas
except Exception:
    pdf_canvas = None


# ---------- MISSING CONSTANTS (ADDED FOR RUNNABILITY) ----------

QUOTES = [
    "You are the only one who can limit your greatness. — Unknown",
    "I have chosen to be happy because it is good for my health. — Voltaire",
    "A sad soul can kill you quicker, far quicker than a germ. — John Steinbeck",
    "The groundwork for all happiness is health. — Leigh Hunt",
    "A calm mind brings inner strength and self-confidence. — Dalai Lama"
]

# Map mood score (1-11) to an emoji and description
MOOD_EMOJI_MAP = {
    1: "😭 Agonizing", 2: "😩 Miserable", 3: "😞 Very Sad",
    4: "🙁 Sad", 5: "😐 Neutral/Okay", 6: "🙂 Content",
    7: "😊 Happy", 8: "😁 Very Happy", 9: "🤩 Excited",
    10: "🥳 Joyful", 11: "🌟 Fantastic"
}

# Define badge rules (functions that return True if the condition is met)
BADGE_RULES = [
    ("First Log", lambda s: len(s["mood_history"]) >= 1),
    ("3-Day Streak", lambda s: s["streaks"].get("mood_log", 0) >= 3),
    ("Consistent Logger", lambda s: len(s["mood_history"]) >= 10),
    ("High Roller", lambda s: any(e.get("mood", 0) >= 10 for e in s["mood_history"])),
    ("Self-Aware", lambda s: len(s["mood_history"]) >= 5 and s["streaks"].get("mood_log", 0) >= 5)
]


# ---------- PERFORMANCE OPTIMIZATIONS (Caching) ----------

@st.cache_resource(show_spinner="Setting up AI connection...")
def setup_ai_model(api_key):
    """Configures and returns the Gemini model, running only once."""
    if not api_key or not genai:
        return None, False
    try:
        genai.configure(api_key=api_key)
        # Using gemini-2.5-flash for faster chat/text tasks
        model = genai.GenerativeModel("gemini-2.5-flash") 
        # Optional: Test a quick call to ensure connection
        # _ = model.generate_content("hello", max_output_tokens=5) 
        return model, True
    except Exception:
        return None, False

@st.cache_resource(show_spinner="Connecting to Database...")
def setup_supabase_client(url, key):
    """Creates and returns the Supabase client, running only once."""
    if not url or not key or not create_client:
        return None, False
    try:
        supabase_client = create_client(url, key)
        return supabase_client, True
    except Exception:
        return None, False

@st.cache_resource
def setup_analyzer():
    """Initializes and returns the VADER sentiment analyzer, running only once."""
    return SentimentIntensityAnalyzer()

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

# ---------- SERVICES CALLS (Using Cached Functions) ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
model, ai_available = setup_ai_model(GEMINI_API_KEY)

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
supabase, db_connected = setup_supabase_client(SUPABASE_URL, SUPABASE_KEY)

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

analyzer = setup_analyzer() # Using the cached function

# ---------- HELPERS ----------
def now_ts(): return time.time()

def clean_text_for_ai(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300) -> str:
    prompt_clean = clean_text_for_ai(prompt)
    if ai_available and model:
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

# ---------- Supabase helpers (guarded & cached) ----------
# Register and Get are typically not cached as they are authentication-related

def register_user_db(email: str):
    if not db_connected or supabase is None: return None
    try:
        res = supabase.table("users").insert({"email": email}).execute()
        if getattr(res, "data", None):
            return res.data[0].get("id")
    except Exception as e:
        # In a real app, this should log to a service, not show in a warning
        # st.warning(f"Supabase register failed: {e}") 
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
        # Clear cache for journal entries on save
        load_journal_db.clear() 
        supabase.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception as e:
        st.warning(f"Save to DB failed: {e}")
        return False

@st.cache_data(show_spinner="Loading journal data...")
def load_journal_db(user_id, supabase_client):
    """Loads all journal entries for a user, only re-running if user_id changes or cache is cleared."""
    if not supabase_client: return []
    try:
        # Use the cached client
        res = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
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
                    # Check if user exists
                    user_list = get_user_by_email_db(email)
                    if user_list:
                        user = user_list[0]
                
                if user:
                    # Login (User found)
                    st.session_state["user_id"] = user.get("id")
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    # Load data using the cached function
                    entries = load_journal_db(st.session_state["user_id"], supabase) or []
                    st.session_state["daily_journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in entries]
                    st.sidebar.success("Logged in.")
                    st.rerun()
                else:
                    # Register (User not found)
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
                        # Local Login
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
            # Clear all session data on logout (including user-specific caches)
            st.session_state.clear()
            st.session_state["page"] = "Home"
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
                st.session_state["page"] = "AI Chat"
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
            
            # Streak calculation logic
            last_dt = None
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
                except Exception:
                    pass

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
                    # Pass the state components needed by the rule
                    state_subset = {"mood_history": st.session_state["mood_history"], "streaks": st.session_state["streaks"]}
                    if rule(state_subset):
                        if name not in st.session_state["streaks"]["badges"]:
                            st.session_state["streaks"]["badges"].append(name)
                except Exception:
                    continue
            st.rerun()

    with col2:
        st.subheader("Badges")
        # Display badges using Streamlit's native st.badge for better style
        for b in st.session_state["streaks"]["badges"]:
            st.badge(b, color="yellow")
            
        st.subheader("Streak")
        st.markdown(f"Consecutive days logging mood: **{st.session_state['streaks'].get('mood_log',0)}**")

    if st.session_state["mood_history"]:
        df = pd.DataFrame(st.session_state["mood_history"]).copy()
        df['date'] = pd.to_datetime(df['date'])
        fig = px.line(df, x='date', y='mood', title="Mood Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def ai_chat_panel():
    st.header("AI Chat")
    st.markdown("A compassionate AI buddy to listen.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hello, I'm here to listen. What's on your mind today?"}]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_response = safe_generate(prompt)
                
                st.markdown(ai_response)
                # Ensure the full content is logged, not just the placeholder
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
        st.rerun()

def mindful_breathing_panel():
    st.header("Mindful Breathing")
    st.markdown("Follow the prompts: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Try 3 cycles.")
    
    # Initialize state variables for the timer
    if "breathing_running" not in st.session_state:
        st.session_state.breathing_running = False
        st.session_state.breathing_cycle = 0
        st.session_state.breathing_start_time = None

    if st.button("Start Exercise", key="start_breathing_btn") and not st.session_state.breathing_running:
        st.session_state.breathing_running = True
        st.session_state.breathing_cycle = 0
        st.session_state.breathing_start_time = time.time()
        st.rerun() # Start the loop

    if st.session_state.breathing_running:
        
        # Define phases: (Phase Name, Duration in Seconds, Color)
        PHASES = [
            ("Inhale", 4.0, "#4a90e2"), # Blue
            ("Hold", 4.0, "#357bd9"),    # Darker Blue
            ("Exhale", 6.0, "#f39c12")   # Orange
        ]

        total_cycle_time = sum(p[1] for p in PHASES)
        elapsed_time = time.time() - st.session_state.breathing_start_time
        
        # Calculate current cycle and time within that cycle
        cycle_number = int(elapsed_time // total_cycle_time) + 1
        current_time_in_cycle = elapsed_time % total_cycle_time
        
        if cycle_number > 3:
            st.session_state.breathing_running = False
            st.session_state.breathing_cycle = 3
            st.success("Exercise complete! You did a great job.")
            return

        st.info(f"Cycle {cycle_number} of 3")
        placeholder_text = st.empty()
        placeholder_progress = st.empty()
        
        current_phase_time = 0.0
        found_phase = False
        for phase, duration, color in PHASES:
            if current_time_in_cycle < current_phase_time + duration:
                # We are in this phase
                time_in_phase = current_time_in_cycle - current_phase_time
                progress_in_phase = time_in_phase / duration
                
                # Display the current phase
                placeholder_text.markdown(
                    f"<h2 style='text-align: center; color: {color}; font-size: 2.5em;'>{phase}</h2>", 
                    unsafe_allow_html=True
                )
                
                # Progress bar display
                placeholder_progress.progress(progress_in_phase)
                
                found_phase = True
                
                # Use a short sleep and rerun to update the bar without freezing the UI
                time.sleep(0.1) 
                st.rerun() 
                break # Exit the loop once the current phase is found
            
            current_phase_time += duration
        
        if not found_phase:
            # Should not happen if logic is correct, but a fallback
            time.sleep(0.1) 
            st.rerun()
            
    if not st.session_state.breathing_running and st.session_state.breathing_cycle < 3:
        if st.button("Reset", key="reset_breathing_btn"):
            st.session_state.breathing_cycle = 0
            st.session_state.breathing_start_time = None
            st.rerun()
            
def mindful_journaling_panel():
    st.header("Mindful Journaling")
    st.markdown("Write freely — your words are private here unless you save to your account.")
    journal_text = st.text_area("Today's reflection", height=220, key="journal_text")
    if st.button("Save Entry"):
        if journal_text.strip():
            sent = sentiment_compound(journal_text)
            
            # Check if logged in and connected to DB
            if st.session_state.get("logged_in") and db_connected and st.session_state.get("user_id"):
                ok = save_journal_db(st.session_state.get("user_id"), journal_text, sent)
                if ok:
                    st.success("Saved to your account.")
                else:
                    st.warning("Could not save to DB. Saved locally instead.")
                    st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
            else:
                # Save locally
                st.session_state["daily_journal"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": journal_text, "sentiment": sent})
                st.success("Saved locally.")
                
            st.rerun()
        else:
            st.warning("Write something you want to save.")

def journal_analysis_panel():
    st.header("Journal & Analysis")
    all_text = get_all_user_text()
    if not all_text:
        st.info("No journal or chat text yet — start journaling or talking to get insights.")
        return
    
    entries = []
    # Journal entries
    for e in st.session_state["daily_journal"]:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e.get("sentiment",0)})
    
    # User chat entries
    chat_entries = [{"date": datetime.now(), "compound": sentiment_compound(msg["content"])} for msg in st.session_state.chat_messages if msg["role"] == "user"]
    entries.extend(chat_entries)

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
    if ai_available and model:
        try:
            # Use safe_generate for a consistent response wrapper
            story = safe_generate(prompt, max_tokens=500)
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
    # Note: st.session_state["call_history"] is always empty in your provided code; 
    # it seems intended for a voice/call feature not fully implemented. Using chat_messages instead.
    for msg in st.session_state.chat_messages:
        if msg.get("role") == "user":
            entries.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": msg.get("content"),
                "sentiment": sentiment_compound(msg.get("content", "")),
            })

    df = pd.DataFrame(entries)

    pos = len(df[df["sentiment"] >= 0.05]) if not df.empty else 0
    neg = len(df[df["sentiment"] <= -0.05]) if not df.empty else 0
    neu = len(df) - pos - neg if not df.empty else 0

    st.subheader("Sentiment Breakdown")
    st.write(f"- Positive entries: {pos}")
    st.write(f"- Neutral entries: {neu}")
    st.write(f"- Negative entries: {neg}")

    summary = "Summary unavailable due to AI error."
    if ai_available and model:
        try:
            summary = safe_generate(
                f"Summarize this person’s emotional trends in a supportive way:\n\n{all_text[:4000]}", max_tokens=300
            )
        except Exception:
            pass
    elif not ai_available:
        summary = "Based on your recent entries, you’re showing resilience and self-awareness. Keep going!"
    
    st.subheader("AI Summary")
    st.markdown(summary)

    if st.button("Export as PDF"):
        if pdf_canvas:
            buffer = io.BytesIO()
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            
            y_position = 750
            c.drawString(40, y_position, "Personalized Wellness Report")
            y_position -= 20
            c.line(40, y_position, 570, y_position)
            y_position -= 20

            # Split summary into lines to fit on PDF
            lines = summary.split('\n')
            for line in lines:
                c.drawString(40, y_position, line)
                y_position -= 15
                if y_position < 50: # Check if near bottom of page
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = 750

            c.save()
            st.download_button(
                label="Download Report PDF",
                data=buffer.getvalue(),
                file_name="wellness_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("PDF generation library (reportlab) is not installed or failed to import.")

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
        "AI Chat": ai_chat_panel,
        "Mindful Breathing": mindful_breathing_panel,
        "Mindful Journaling": mindful_journaling_panel,
        "Journal & Analysis": journal_analysis_panel,
        "My Emotional Journey": emotional_journey_panel,
        "Personalized Report": personalized_report_panel,
        "Crisis Support": crisis_support_panel
    }
    
    page_names = list(pages.keys())
    
    # Initialize session state for the page if it's not present
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