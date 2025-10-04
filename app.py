import streamlit as st
import os
import time
import random
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np

# Lightweight sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import the OpenAI library (used for OpenRouter compatibility)
from openai import OpenAI
from openai import APIError

# Placeholder for Supabase client
from supabase import create_client

# ---------- CONSTANTS ----------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" # Or any other OpenRouter model
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
    ("Breathing Master", lambda s: "Breathing Master" in s["streaks"]["badges"]),
    ("Self-Aware", lambda s: len(s["mood_history"]) >= 5 and s["streaks"].get("mood_log", 0) >= 5)
]

PHQ9_QUESTIONS = [
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?",
    "3. Trouble falling or staying asleep, or sleeping too much?",
    "4. Feeling tired or having little energy?",
    "5. Poor appetite or overeating?",
    "6. Feeling bad about yourself—or that you are a failure or have let yourself or your family down?",
    "7. Trouble concentrating on things, such as reading the newspaper or watching television?",
    "8. Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?",
    "9. Thoughts that you would be better off dead or of hurting yourself in some way?"
]

PHQ9_SCORES = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

PHQ9_INTERPRETATION = {
    (0, 4): "Minimal to None",
    (5, 9): "Mild",
    (10, 14): "Moderate",
    (15, 19): "Moderately Severe",
    (20, 27): "Severe"
}


# ---------- Streamlit page config and LAYOUT SETUP ----------
st.set_page_config(
    page_title="AI Wellness Companion", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_page_and_layout():
    # --- DEFINITIVE CSS INJECTION TO FIX TEXT AREA ---
    # This aggressive CSS block ensures the text area is always visible and functional.
    st.markdown("""
    <style>
    /* 1. Global Background and Typography */
    .stApp { 
        background: #F0F2F6; /* Very Light Grey/Off-White background */
        color: #1E1E1E; 
        font-family: 'Poppins', sans-serif; 
    }
    .main .block-container { 
        padding: 2rem 4rem; /* Generous padding */
    }
    
    /* 2. CRITICAL: Target the Streamlit Text Area's internal input element */
    textarea {
        color: black !important; /* Forces the input text to be black */
        -webkit-text-fill-color: black !important; /* For Safari/Chrome text color issues */
        opacity: 1 !important; /* Ensure it's not hidden by opacity */
        background-color: white !important; /* Ensure background is white */
        border: 1px solid #ccc !important; /* Ensure a visible border */
    }

    /* 3. Target the main Streamlit container for the text area */
    .stTextArea > div > div > textarea {
        color: black !important;
    }

    /* 4. Ensure the div wrapping the text area is visible */
    .stTextArea {
        opacity: 1 !important; 
    }
    
    /* 5. Sidebar Aesthetics */
    .css-1d3f90z { /* Streamlit sidebar selector */
        background-color: #FFFFFF; /* White background for sidebar */
        box-shadow: 2px 0 10px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    
    /* 6. Custom Card Style (The Core Mobile App Look) */
    .card { 
        background-color: #FFFFFF; /* White background for the card */
        border-radius: 16px; /* Highly rounded corners */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); /* Soft, noticeable shadow for lift */
        padding: 25px; /* Slightly more padding */
        margin-bottom: 20px; 
        border: none; 
        transition: all .2s ease-in-out; 
    }
    .card:hover { 
        transform: translateY(-3px); /* Subtle hover lift */
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15); 
    }

    /* 7. Primary Button Style (Vibrant and Rounded) */
    .stButton>button { 
        color: #FFFFFF; 
        background: #5D54A4; /* Deep Purple */
        border-radius: 25px; /* Pill shape */
        padding: 10px 20px; 
        font-weight: 600; 
        border: none; 
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        transition: all .2s;
    }
    .stButton>button:hover { 
        background: #7A72BF;
    }
    
    /* 8. Custom Sidebar Status */
    .sidebar-status {
        padding: 5px 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    .status-connected { background-color: #D4EDDA; color: #155724; border-left: 4px solid #28A745; }
    .status-local { background-color: #FFEEDD; color: #856404; border-left: 4px solid #FFC107; }
    
    /* 9. Larger Titles for Impact */
    h1 { color: #1E1E1E; font-weight: 700; margin-bottom: 0.5rem; }
    h2 { color: #333333; font-weight: 600; margin-top: 2rem;}
    h3 { color: #5D54A4; font-weight: 500; margin-top: 1rem;}

    /* 10. Breathing Circle Styles */
    .breathing-circle {
        width: 100px;
        height: 100px;
        background-color: #5D54A4;
        border-radius: 50%;
        margin: 50px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        transition: all 4s ease-in-out;
    }
    .breathe-inhale {
        animation: scaleIn 4s infinite alternate;
    }
    .breathe-exhale {
        animation: scaleOut 6s infinite alternate;
    }
    @keyframes scaleIn {
        from { transform: scale(1); }
        to { transform: scale(2.5); }
    }
    @keyframes scaleOut {
        from { transform: scale(2.5); }
        to { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Call the setup function early in the main script flow
setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------

@st.cache_data
def initialize_kalman(Q_val=0.01, R_val=0.1):
    """Initializes the Kalman filter state variables."""
    return {
        'x_est': 75.0,  # Estimated state (Heart Rate)
        'P_est': 1.0,   # Estimated covariance
        'Q': Q_val,     # Process noise covariance
        'R': R_val      # Measurement noise covariance
    }

def kalman_filter_simple(z_meas, state):
    """Applies a single step of the Kalman filter to a noisy measurement."""
    
    # 1. Prediction
    x_pred = state['x_est'] 
    P_pred = state['P_est'] + state['Q']

    # 2. Update
    K = P_pred / (P_pred + state['R'])
    x_est = x_pred + K * (z_meas - x_pred)
    P_est = (1 - K) * P_pred

    # Update state dictionary
    state['x_est'] = x_est
    state['P_est'] = P_est
    return x_est, state

def generate_simulated_physiological_data(current_time_ms):
    """
    Simulates noisy PPG (Heart Rate) and GSR (Stress) data.
    """
    
    time_sec = current_time_ms / 1000.0 
    
    # Base HR (BPM) that gently changes over time (70-100 BPM)
    base_hr = 85 + 10 * np.sin(time_sec / 30.0) 
    
    # Add high-frequency noise (Simulates a noisy sensor/motion)
    ppg_noise = 3 * random.gauss(0, 1)
    
    # Simulate Filtered HR (The 'clean' signal we *want* to see)
    clean_hr = base_hr + 2 * np.sin(time_sec / 0.5) 
    
    # Raw PPG Measurement (Noisy sine wave that simulates a pulse)
    raw_ppg_signal = clean_hr + ppg_noise
    
    # GSR/Stress Simulation (correlated with base HR and overall phq9 score)
    base_gsr = 0.5 * base_hr / 100.0
    phq9_score = st.session_state.get("phq9_score") or 0
    # Normalize score by max possible (27)
    gsr_value = 1.0 + base_gsr + 0.5 * np.random.rand() * (phq9_score / 27.0)
    
    return {
        "raw_ppg_signal": raw_ppg_signal, 
        "gsr_stress_level": gsr_value,
        "time_ms": current_time_ms
    }

# ---------- CACHING & LAZY SETUP ----------
@st.cache_resource
def setup_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def setup_ai_model(api_key: str, history: list):
    """Lazy configure OpenAI client for OpenRouter."""
    if not api_key:
        return None, False, history # Return existing history if setup fails
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL
        )
        
        system_instruction = """
You are 'The Youth Wellness Buddy,' an AI designed for teens. 
Your primary goal is to provide non-judgemental, empathetic, and encouraging support. 
Your personality is warm, slightly informal, and very supportive.
"""
        # Append system instruction to history if it's the first element
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": system_instruction})
        
        if len(history) <= 1:
             history.append({"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"})

        return client, True, history
    except Exception:
        return None, False, history


@st.cache_resource(show_spinner=False)
def setup_supabase_client(url: str, key: str):
    if not url or not key:
        return None, False
    try:
        client = create_client(url, key)
        return client, True
    except Exception:
        return None, False

# ---------- Session state defaults (CLEANED UP) ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# IoT/ECE State
if "kalman_state" not in st.session_state:
    st.session_state["kalman_state"] = initialize_kalman()
if "physiological_data" not in st.session_state:
    st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])

# AI/DB/Auth State
if "_ai_model" not in st.session_state:
    raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
    
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    
    _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY, st.session_state["chat_messages"])
    st.session_state["_ai_model"] = _ai_client_obj 
    st.session_state["_ai_available"] = _ai_available
    st.session_state["chat_messages"] = _chat_history_list
    
if "_supabase_client_obj" not in st.session_state:
    raw_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    raw_key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

    SUPABASE_URL = raw_url.strip().strip('"') if isinstance(raw_url, str) and raw_url else None
    SUPABASE_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
    
    _supabase_client_obj, _db_connected = setup_supabase_client(SUPABASE_URL, SUPABASE_KEY)
    st.session_state["_supabase_client_obj"] = _supabase_client_obj
    st.session_state["_db_connected"] = _db_connected

if "daily_journal" not in st.session_state:
    st.session_state["daily_journal"] = []

if "mood_history" not in st.session_state:
    st.session_state["mood_history"] = []

if "streaks" not in st.session_state:
    st.session_state["streaks"] = {"mood_log": 0, "last_mood_date": None, "badges": []}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

# PHQ-9 State
if "phq9_score" not in st.session_state:
    st.session_state["phq9_score"] = None

if "phq9_interpretation" not in st.session_state:
    st.session_state["phq9_interpretation"] = None

if "last_phq9_date" not in st.session_state:
    st.session_state["last_phq9_date"] = None
    
# CBT/Journaling State
if "last_reframing_card" not in st.session_state:
    st.session_state["last_reframing_card"] = None

# Breathing State
if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop" # 'stop', 'running', 'finished'

analyzer = setup_analyzer()

# ---------- AI/Sentiment Helper functions ----------
def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300):
    """
    Generate text via OpenRouter, with system message and custom fallback.
    """
    prompt_lower = prompt.lower()
    
    # Custom, empathetic responses for key phrases
    if any(phrase in prompt_lower for phrase in ["demotivated", "heavy", "don't want to do anything", "feeling down"]):
        return (
            "Thanks for sharing that with me. That feeling of demotivation can be really heavy, and it takes a lot of courage just to name it. I want you to know you're not alone. Before we try to tackle the whole mountain, let's just look at one rock. Is there one tiny task or thought that feels the heaviest right now? 🌱"
        )
    
    # Default AI generation
    if st.session_state.get("_ai_available") and st.session_state.get("_ai_model"):
        client = st.session_state["_ai_model"]
        messages_for_api = st.session_state.chat_messages
        prompt_clean = clean_text_for_ai(prompt)

        # Append new user message before sending to API
        if messages_for_api[-1]["content"] != prompt_clean or messages_for_api[-1]["role"] != "user":
             messages_for_api.append({"role": "user", "content": prompt_clean})

        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=messages_for_api,
                max_tokens=max_tokens,
                temperature=0.7 
            )
            
            if resp.choices and resp.choices[0].message:
                return resp.choices[0].message.content
            
        except APIError:
             st.error("OpenRouter API Error. Please check your key or try a different model.")
        except Exception:
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
    return " ".join(parts).strip()


# ---------- Supabase helpers (Retained) ----------
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


# ---------- Sidebar Navigation and Auth ----------

# Sidebar Status
st.sidebar.markdown("### Status")
ai_status_class = "status-connected" if st.session_state.get("_ai_available") else "status-local"
db_status_class = "status-connected" if st.session_state.get("_db_connected") else "status-local"

st.sidebar.markdown(
    f"<div class='sidebar-status {ai_status_class}'>AI: <b>{'Connected (OpenRouter)' if st.session_state.get('_ai_available') else 'Local (fallback)'}</b></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<div class='sidebar-status {db_status_class}'>DB: <b>{'Connected' if st.session_state.get('_db_connected') else 'Not connected'}</b></div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.header("Navigation")
page_options = {
    "Home": "🏠", 
    "AI Chat": "💬", 
    "Mood Tracker": "📈", 
    "Mindful Journaling": "📝", 
    "Journal Analysis": "📊",
    "Mindful Breathing": "🧘‍♀️", 
    "IoT Dashboard (ECE)": "⚙️", 
    "Wellness Check-in": "🩺",
    "Report & Summary": "📄"
}

st.session_state["page"] = st.sidebar.radio(
    "Go to:", 
    list(page_options.keys()), 
    format_func=lambda x: f"{page_options[x]} {x}",
    key="sidebar_navigation"
)


# Sidebar Auth
def sidebar_auth():
    st.sidebar.markdown("---")
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
                
                if user or st.session_state.get("_db_connected") is False:
                    st.session_state["user_id"] = user.get("id") if user else "local_user"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    st.session_state["daily_journal"] = [] 
                    
                    if user and st.session_state.get("_db_connected"):
                        entries = load_journal_db(st.session_state["user_id"], st.session_state.get("_supabase_client_obj")) or []
                        st.session_state["daily_journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in entries]
                        st.sidebar.success("Logged in and data loaded. ✅")
                    elif st.session_state.get("_db_connected") is False:
                         st.sidebar.info("Logged in locally (no DB). 🏠")
                         
                    st.rerun()

                else:
                    uid = register_user_db(email)
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.sidebar.success("Registered & logged in. 🎉")
                        st.rerun()
                    else:
                        st.sidebar.error("Registration failed. Try again or check DB connection.")
            else:
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state"]:
                if key in st.session_state:
                    st.session_state[key] = None
            
            # Reset major states
            st.session_state["daily_journal"] = []
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            
            # Reset AI history
            raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
            _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY, [])
            st.session_state["_ai_model"] = _ai_client_obj
            st.session_state["_ai_available"] = _ai_available
            st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]

            st.sidebar.info("Logged out. 👋")
            st.rerun()

sidebar_auth()


# ---------- PANELS: Homepage ----------
def homepage_panel():
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #5D54A4;'>🧠</span></h1>", unsafe_allow_html=True)
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    
    st.markdown("---")
    
    # --- Row 1: Daily Inspiration Card ---
    with st.container():
        st.markdown("<div class='card' style='border-left: 8px solid #FFC107;'>", unsafe_allow_html=True)
        st.markdown("<h3>Daily Inspiration ✨</h3>")
        st.markdown(f"**<span style='font-size: 1.25rem; font-style: italic;'>“{random.choice(QUOTES)}”</span>**", unsafe_allow_html=True)
        st.markdown("<p style='text-align: right; margin-top: 10px; font-size: 0.9rem;'>— Take a moment for yourself</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- Row 2: Quick Actions ---
    st.markdown("<h2>Quick Actions</h2>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("🧘‍♀️ Breathe", key="home_breath_btn", use_container_width=True):
            st.session_state["page"] = "Mindful Breathing"
            st.rerun()
    with c2:
        if st.button("💬 Chat AI", key="home_chat_btn", use_container_width=True):
            st.session_state["page"] = "AI Chat"
            st.rerun()
    with c3:
        if st.button("📝 Journal", key="home_journal_btn", use_container_width=True):
            st.session_state["page"] = "Mindful Journaling"
            st.rerun()
    with c4:
        if st.button("🩺 Check-in", key="home_checkin_btn", use_container_width=True):
            st.session_state["page"] = "Wellness Check-in"
            st.rerun()
            
    st.markdown("---")
    
    # --- Row 3: Feature Cards ---
    st.markdown("<h2>Your Toolkit</h2>", unsafe_allow_html=True)
    
    col_mood, col_journal = st.columns(2)

    with col_mood:
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #5D54A4; height: 180px;'>
            <h3 style='margin-top:0;'>Mood Tracker & Analysis 📈</h3>
            <p style='font-size: 0.95rem;'>Log your daily emotional state and see your personal timeline evolve. Earn badges for consistency.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\\'📈 Mood Tracker\\']').click(); return false;" style='color:#5D54A4; font-weight: 600; text-decoration: none;'>View Dashboard →</a></div>
        </div>
        """, unsafe_allow_html=True)

    with col_journal:
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #28A745; height: 180px;'>
            <h3 style='margin-top:0;'>Mindful Journaling 📝</h3>
            <p style='font-size: 0.95rem;'>A private space for reflection. The AI analyzes your entries to provide insights and sentiment scores.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\\'📝 Mindful Journaling\\']').click(); return false;" style='color:#28A745; font-weight: 600; text-decoration: none;'>Start Writing →</a></div>
        </div>
        """, unsafe_allow_html=True)
        
    # IoT/ECE Card
    with st.container():
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #FF5733;'>
            <h3 style='margin-top:0;'>IoT Monitoring (ECE Demo) ⚙️</h3>
            <p style='font-size: 0.95rem;'>Simulated real-time physiological data (Heart Rate/Stress) using a Kalman Filter demonstration.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\\'⚙️ IoT Dashboard (ECE)\\']').click(); return false;" style='color:#FF5733; font-weight: 600; text-decoration: none;'>Go to Dashboard →</a></div>
        </div>
        """, unsafe_allow_html=True)


# ---------- PANELS: Mood Tracker ----------
def mood_tracker_panel():
    st.header("Daily Mood Tracker 📈")
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3,1])
        with col1:
            mood = st.slider("How do you feel right now? (1-11)", 1, 11, 6)
            st.markdown(f"**You chose:** {MOOD_EMOJI_MAP.get(mood, 'N/A')} · **{mood}/11**")
            note = st.text_input("Optional: Add a short note about why you feel this way", key="mood_note_input")
            if st.button("Log Mood", key="log_mood_btn"):
                entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
                st.session_state["mood_history"].append(entry)

                # Streak Logic
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

                st.success("Mood logged. Tiny step, big impact. ✨")

                # Badge check
                for name, rule in BADGE_RULES:
                    try:
                        state_subset = {"mood_history": st.session_state["mood_history"], "streaks": st.session_state["streaks"]}
                        if rule(state_subset):
                            if name not in st.session_state["streaks"]["badges"]:
                                st.session_state["streaks"]["badges"].append(name)
                                st.balloons()
                    except Exception:
                        continue
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")

    col_stats, col_badges = st.columns([2,1])

    with col_stats:
        # Plot mood history
        st.subheader("Your Emotional Timeline")
        if st.session_state["mood_history"]:
            df = pd.DataFrame(st.session_state["mood_history"]).copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date') 
            fig = px.line(df, x='date', y='mood', title="Mood Over Time", markers=True,
                          color_discrete_sequence=['#4a90e2'])
            fig.update_layout(yaxis_range=[1, 11])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log your first mood to see your personal timeline!")
    
    with col_badges:
        st.subheader("Current Streak 🔥")
        st.metric("Days", f"{st.session_state['streaks'].get('mood_log',0)}", delta_color="off")
        
        st.subheader("Badges Earned 🎖️")
        if st.session_state["streaks"]["badges"]:
            for b in st.session_state["streaks"]["badges"]:
                st.markdown(f"<div style='padding: 8px; margin-bottom: 5px; background-color: #f0f7ff; border-radius: 8px;'>**{b}** 🌟</div>", unsafe_allow_html=True)
        else:
            st.markdown("_No badges yet — log a mood to get started!_")


# ---------- PANELS: AI Chat ----------
def ai_chat_panel():
    st.header("AI Chat 💬")
    st.markdown("Your compassionate AI buddy. Start chatting, and I’ll listen.")
    
    # Status check
    if not st.session_state.get("_ai_available"):
        st.warning("AI is running in 'Local (fallback)' mode. Responses may be simple.")

    with st.container(height=500, border=True):
        # Display chat messages from history
        for message in st.session_state.chat_messages:
            if message["role"] in ["user", "assistant"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your wellness buddy..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.spinner("Thinking..."):
            response = safe_generate(prompt)

        # Append assistant message to history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)

# ---------- PANELS: Mindful Journaling ----------
def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    st.markdown("Use this private space to reflect on your day, thoughts, and emotions.")
    
    ENTRY_KEY = "journal_input_final"
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # --- TEXT AREA ---
        entry_text = st.text_area(
            "What's on your mind right now? (The more you write, the better the analysis!)", 
            height=250, 
            key=ENTRY_KEY,
            value="" 
        )
        # ---------------------------
        
        if st.button("Submit Entry & Analyze", key="submit_journal_btn", use_container_width=True):
            if entry_text:
                sentiment_score = sentiment_compound(entry_text)
                
                new_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    "text": entry_text, 
                    "sentiment": sentiment_score
                }
                st.session_state["daily_journal"].append(new_entry)
                
                # Save to Supabase
                if st.session_state.get("logged_in") and st.session_state.get("_db_connected"):
                    save_journal_db(st.session_state["user_id"], entry_text, sentiment_score)
                    st.success("Entry saved to Supabase! 👍")
                else:
                    st.info("Entry saved locally. Log in to save permanently.")
                
                # --- CBT Enhancement: Simple AI Reframing ---
                if st.session_state.get("_ai_available") and sentiment_score < -0.2:
                    with st.spinner("AI is generating a helpful reframing thought..."):
                        cbt_prompt = f"The user just wrote a journal entry with a negative sentiment (-0.2 or lower). Generate one single, brief, non-clinical 'Coping Card' thought (cognitive reframing statement) based on the user's entry that is empathetic, validates their feeling, and offers a slightly more constructive perspective. The entry was: '{entry_text[:200]}...'"
                        reframing_thought = safe_generate(cbt_prompt, max_tokens=100)
                        
                        st.session_state["last_reframing_card"] = reframing_thought
                
                # FIX: Removed st.session_state[ENTRY_KEY] = ""
                # The page reruns, which naturally resets the widget's value.
                
                st.session_state["page"] = "Journal Analysis"
                st.rerun()
            else:
                st.warning("Please write something before submitting.")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("Previous Entries")
    if st.session_state["daily_journal"]:
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        
        for index, row in df_journal.sort_values(by="date", ascending=False).iterrows():
            date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d %H:%M")
            sentiment_color = "red" if row["sentiment"] < -0.1 else ("green" if row["sentiment"] > 0.1 else "gray")
            
            st.markdown(f"""
            <div class='card' style='padding: 15px; border-left: 5px solid {sentiment_color};'>
                <p style='font-size: 0.8rem; color: #777;'>{date_str}</p>
                <p style='font-size: 1rem; margin-top: 5px;'>{row["text"]}</p>
                <p style='font-size: 0.8rem; color: #5D54A4; text-align: right;'>Sentiment: <b>{row["sentiment"]:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Your journal is empty. Write your first entry!")

# ---------- PANELS: Journal Analysis ----------
def journal_analysis_panel():
    st.header("Journal Analysis 📊")
    
    # 1. CBT Reframing Card (New Feature)
    if st.session_state.get("last_reframing_card"):
        st.markdown(f"""
            <div class='card' style='border-left: 8px solid #FF5733; background-color: #FFF2EB;'>
                <h3 style='margin-top:0; color: #FF5733;'>🧠 Cognitive Reframing Card</h3>
                <p style='font-size: 1.1rem; font-style: italic;'>
                    "{st.session_state["last_reframing_card"]}"
                </p>
                <p style='font-size: 0.9rem; text-align: right; color: #777;'>— Your Wellness Buddy</p>
            </div>
            """, unsafe_allow_html=True)
        st.session_state["last_reframing_card"] = None # Clear after display

    st.markdown("---")
    
    if not st.session_state["daily_journal"]:
        st.info("No journal entries yet. Write your first entry to see the analysis.")
        return

    df_journal = pd.DataFrame(st.session_state["daily_journal"]).copy()
    df_journal['date'] = pd.to_datetime(df_journal['date'])
    df_journal['sentiment_score'] = pd.to_numeric(df_journal['sentiment'], errors='coerce')
    df_journal['word_count'] = df_journal['text'].apply(lambda x: len(str(x).split()))
    
    # Stats row
    col_avg, col_total = st.columns(2)
    with col_avg:
        avg_sentiment = df_journal['sentiment_score'].mean()
        delta_color = "inverse" if avg_sentiment < 0 else "normal"
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}", delta_color=delta_color)
    with col_total:
        total_words = df_journal['word_count'].sum()
        st.metric("Total Words Analyzed", f"{total_words:,}")

    # Sentiment Plot (UPGRADED TO PLOTLY)
    st.subheader("Sentiment Over Time")
    fig = px.bar(
        df_journal, 
        x='date', 
        y='sentiment_score', 
        title="Journal Sentiment History (VADER Compound Score)",
        color='sentiment_score',
        color_continuous_scale=px.colors.sequential.Sunsetdark, # Nice gradient color scheme
        labels={'sentiment_score': 'Sentiment Score', 'date': 'Date'}
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- PANELS: Mindful Breathing ----------
def mindful_breathing_panel():
    st.header("Mindful Breathing 🧘‍♀️")
    st.markdown("Follow the circle for a calming 4-4-6 breathing exercise.")
    
    if st.session_state["breathing_state"] == "stop":
        st.session_state["breathing_state"] = "finished" # Use finished as the initial state
        
    if st.session_state["breathing_state"] == "finished":
        st.subheader("Ready to begin?")
        st.info("The cycle is 4 seconds INHALE, 4 seconds HOLD, 6 seconds EXHALE.")
        if st.button("Start 3-Minute Session", key="start_breathing_btn", use_container_width=True):
            st.session_state["breathing_state"] = "running"
            st.session_state["breathing_start_time"] = time.time()
            st.session_state["breathing_step"] = "INHALE"
            st.session_state["breathing_duration"] = 4
            st.session_state["breathing_progress"] = 0
            st.rerun()

    if st.session_state["breathing_state"] == "running":
        # Check if 3 minutes are up (180 seconds)
        elapsed_time = time.time() - st.session_state["breathing_start_time"]
        if elapsed_time > 180:
            st.session_state["breathing_state"] = "finished"
            st.success("Session complete! You breathed for 3 minutes. Well done! 🏅")
            if "Breathing Master" not in st.session_state["streaks"]["badges"]:
                 st.session_state["streaks"]["badges"].append("Breathing Master")
                 st.balloons()
            st.rerun()
        
        # Determine current phase and duration
        step = st.session_state["breathing_step"]
        duration = st.session_state["breathing_duration"]
        
        # Update progress bar and step logic
        # Increment by a small value to simulate continuous change
        st.session_state["breathing_progress"] += 0.1
        progress_val = st.session_state["breathing_progress"]
        
        # Check if current step is complete
        if progress_val > duration:
            progress_val = 0.1 # Reset counter for next step
            
            if step == "INHALE":
                st.session_state["breathing_step"] = "HOLD"
                st.session_state["breathing_duration"] = 4
            elif step == "HOLD":
                st.session_state["breathing_step"] = "EXHALE"
                st.session_state["breathing_duration"] = 6
            elif step == "EXHALE":
                st.session_state["breathing_step"] = "INHALE"
                st.session_state["breathing_duration"] = 4
            
            st.session_state["breathing_progress"] = progress_val
            st.rerun() # Rerun to update the step text immediately
        
        st.session_state["breathing_progress"] = progress_val
        
        # Display the animation and text
        col_anim, col_info = st.columns([1, 2])
        
        with col_anim:
            # Animation Class Logic
            animation_class = "breathe-inhale" if step == "INHALE" else ("breathe-exhale" if step == "EXHALE" else "")
            
            # Use style to stop the animation when HOLDING
            style = ""
            if step == "HOLD":
                # Scale to max size during hold
                style = "transform: scale(2.5);" 
            else:
                # Apply continuous animation for inhale/exhale
                style = f"animation-duration: {duration}s; animation-iteration-count: infinite;"

            st.markdown(f"""
                <div class='breathing-circle {animation_class}' style='{style}'>
                     {step}
                </div>
            """, unsafe_allow_html=True)


        with col_info:
            st.subheader(f"Current Phase: {st.session_state['breathing_step']}")
            st.metric("Time Left (Approx.)", f"{int(180 - elapsed_time)} seconds")
            
            # Use st.progress to show the progress within the current step
            progress_percent = st.session_state["breathing_progress"] / st.session_state["breathing_duration"]
            st.progress(progress_percent, text=f"{int(st.session_state['breathing_progress'])}/{st.session_state['breathing_duration']} seconds")

            st.caption("Do not close this window. Focus only on the circle.")
            if st.button("Stop Session", key="stop_breathing_btn"):
                st.session_state["breathing_state"] = "finished"
                st.rerun()
                
        # --- Rerun the app every 100ms for smooth animation/timing ---
        time.sleep(0.1)
        st.rerun()


# ---------- PANELS: IoT Dashboard (ECE Demo) ----------
def iot_dashboard_panel():
    st.header("IoT Dashboard (ECE Demo) ⚙️")
    st.markdown("Simulated real-time physiological data using a **Kalman Filter** for noise reduction.")
    
    # 1. Generate and Filter New Data
    current_time_ms = int(time.time() * 1000)
    new_raw_data = generate_simulated_physiological_data(current_time_ms)
    
    # Apply Kalman Filter to the noisy PPG (Heart Rate) signal
    filtered_hr, st.session_state["kalman_state"] = kalman_filter_simple(
        new_raw_data["raw_ppg_signal"], 
        st.session_state["kalman_state"]
    )
    
    new_data_point = {
        "time_ms": current_time_ms,
        "raw_ppg_signal": new_raw_data["raw_ppg_signal"],
        "filtered_hr": filtered_hr,
        "gsr_stress_level": new_raw_data["gsr_stress_level"]
    }
    
    # 2. Update DataFrame
    df = st.session_state["physiological_data"]
    new_row = pd.DataFrame([new_data_point])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Keep only the last 100 points for real-time window effect
    df = df.tail(100)
    st.session_state["physiological_data"] = df
    
    # 3. Display Metrics
    col_hr, col_stress = st.columns(2)
    
    with col_hr:
        # Use the *Filtered* value for the primary metric
        st.metric(
            "Reliable Heart Rate (BPM)", 
            f"{filtered_hr:.1f}", 
            delta=f"{new_raw_data['raw_ppg_signal'] - filtered_hr:.1f} (Noise Error)", 
            delta_color="off"
        )
    with col_stress:
        # Scale GSR to a visible range
        stress_level = max(0, new_raw_data["gsr_stress_level"] - 1.0) * 10 
        stress_emoji = "🟢 Low" if stress_level < 5 else ("🟡 Medium" if stress_level < 10 else "🔴 High")
        st.metric("GSR Stress Index", f"{stress_level:.1f} {stress_emoji}", delta_color="off")

    st.markdown("---")
    
    # 4. Plotting (Plotly for interactivity)
    st.subheader("Kalman Filter Demonstration (HR/PPG)")
    
    df['Time'] = (df['time_ms'] - df['time_ms'].min()) / 1000.0
    
    # Melt the DataFrame for Plotly express to plot two lines easily
    df_plot = df.melt(id_vars='Time', value_vars=['raw_ppg_signal', 'filtered_hr'],
                      var_name='Signal Type', value_name='BPM')
    
    fig = px.line(df_plot, x='Time', y='BPM', color='Signal Type',
                  title='Raw vs. Filtered Heart Rate Signal',
                  color_discrete_map={
                      'raw_ppg_signal': '#FF4B4B',   # Red for noisy raw
                      'filtered_hr': '#5D54A4'      # Purple for stable filtered
                  })
    fig.update_layout(xaxis_title="Time Elapsed (s)", yaxis_title="Heart Rate (BPM)", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Re-run the app every 500ms for a real-time feel
    time.sleep(0.5)
    st.rerun()


# ---------- PANELS: Wellness Check-in (PHQ-9) ----------
def wellness_checkin_panel():
    st.header("Wellness Check-in 🩺")
    st.markdown("This is a simplified, non-diagnostic version of the PHQ-9. Use it to check in with yourself.")
    
    today_str = datetime.now().strftime("%Y-%m-%d")

    if st.session_state.get("last_phq9_date") == today_str and st.session_state.get("phq9_score") is not None:
        # Already completed today
        st.info(f"You completed your check-in today ({today_str}).")
        st.subheader("Your Score Today")
        score = st.session_state["phq9_score"]
        interpretation = st.session_state["phq9_interpretation"]
        
        color = 'green'
        if score >= 5 and score <= 9: color = 'orange'
        elif score >= 10: color = 'red'
            
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid {color};'>
            <h3 style='margin-top:0;'>Total Score: {score} / 27</h3>
            <p style='font-size: 1.1rem;'>**Interpretation:** {interpretation}</p>
            <p style='font-size: 0.9rem; color: #777;'>*Scores are for self-reflection and not medical advice.*</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # User has not completed today's check-in
    st.subheader("Over the last two weeks, how often have you been bothered by the following?")
    
    phq9_answers = {}
    
    with st.form("phq9_form"):
        for i, question in enumerate(PHQ9_QUESTIONS):
            phq9_answers[i] = st.radio(
                question, 
                list(PHQ9_SCORES.keys()), 
                key=f"phq9_q{i}"
            )
            st.markdown("---")
            
        submit_button = st.form_submit_button("Calculate My Score", use_container_width=True)
        
    if submit_button:
        total_score = 0
        for i in range(len(PHQ9_QUESTIONS)):
            answer = phq9_answers[i]
            total_score += PHQ9_SCORES.get(answer, 0)
            
        # Determine interpretation
        interpretation = "N/A"
        for score_range, text in PHQ9_INTERPRETATION.items():
            if score_range[0] <= total_score <= score_range[1]:
                interpretation = text
                break
        
        # Save results to session state
        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation
        st.session_state["last_phq9_date"] = today_str
        
        # Add to mood history (optional, for data analysis)
        phq9_mood = max(1, 11 - (total_score // 3)) # Maps 0->11, 27->2
        st.session_state["mood_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "mood": phq9_mood, 
            "note": f"PHQ-9 Check-in: Score {total_score} ({interpretation})"
        })

        st.success(f"Check-in complete! Your score is {total_score}.")
        st.balloons()
        st.rerun()


# ---------- PANELS: Report & Summary ----------
def report_summary_panel():
    st.header("Report & Summary 📄")
    st.markdown("A consolidated view of your wellness journey and AI-generated insights.")

    col1, col2, col3, col4 = st.columns(4)

    # 1. Key Metrics Overview
    with col1:
        st.subheader("Mood")
        avg_mood = pd.DataFrame(st.session_state["mood_history"])['mood'].mean() if st.session_state["mood_history"] else 0
        st.metric("Avg. Mood Score", f"{avg_mood:.1f}/11")

    with col2:
        st.subheader("Journal")
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        avg_sentiment = df_journal['sentiment'].mean() if not df_journal.empty else 0
        st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")

    with col3:
        st.subheader("PHQ-9")
        score = st.session_state.get("phq9_score", "N/A")
        if score != "N/A":
             st.metric("Last Check-in Score", f"{score}/27", help=st.session_state.get("phq9_interpretation", ""))
        else:
             st.metric("Last Check-in Score", "N/A")

    with col4:
        st.subheader("Consistency")
        st.metric("Mood Log Streak", f"{st.session_state['streaks'].get('mood_log', 0)} days")

    st.markdown("---")

    # 2. AI Narrative Summary (Actionable Insight)
    st.subheader("Personalized AI Summary")
    
    if st.session_state.get("_ai_available"):
        if st.button("Generate Comprehensive Wellness Summary", use_container_width=True):
            full_text = get_all_user_text()
            
            if len(full_text.split()) < 50:
                st.warning("Please create more journal entries or chat interactions (minimum 50 words) to generate a meaningful summary.")
            else:
                with st.spinner("AI is analyzing all your data points to create your summary..."):
                    # Craft a very specific prompt
                    summary_prompt = f"""
                    Analyze the user's complete text history (journal entries and chat messages) and their key metrics (Avg Mood: {avg_mood:.1f}, Avg Sentiment: {avg_sentiment:.2f}, Mood Streak: {st.session_state['streaks'].get('mood_log', 0)}).

                    Generate a concise, compassionate, 3-paragraph summary that:
                    1. Acknowledges their current state and mood trend (e.g., stable, improving, dipping).
                    2. Identifies a central theme, concern, or source of stress/joy from their text.
                    3. Provides one single, practical, non-clinical recommendation for the next 7 days based on the data.
                    
                    The text history for analysis is: {full_text[:3000]}...
                    """
                    
                    summary_text = safe_generate(summary_prompt, max_tokens=600)
                    
                    st.markdown(f"""
                        <div class='card' style='border-left: 8px solid #5D54A4; background-color: #F0F2FF;'>
                            <p style='white-space: pre-wrap;'>{summary_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
    else:
        st.warning("AI features are not available. Check your OpenRouter API key.")

    st.markdown("---")
    
    # 3. Raw Data Visualizations (Re-use existing plots)
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("Mood Timeline")
        if st.session_state["mood_history"]:
            df = pd.DataFrame(st.session_state["mood_history"]).copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            fig = px.line(df, x='date', y='mood', markers=True, color_discrete_sequence=['#4a90e2'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col_vis2:
        st.subheader("Sentiment Distribution")
        if not df_journal.empty:
            fig = px.histogram(df_journal, x='sentiment', nbins=20, title="Sentiment Distribution", color_discrete_sequence=['#28A745'])
            fig.update_layout(yaxis_title="Number of Entries", height=300)
            st.plotly_chart(fig, use_container_width=True)


# ---------- MAIN ROUTER (RETAINED) ----------
if st.session_state["page"] == "Home":
    homepage_panel()
elif st.session_state["page"] == "AI Chat":
    ai_chat_panel()
elif st.session_state["page"] == "Mood Tracker":
    mood_tracker_panel()
elif st.session_state["page"] == "Mindful Journaling":
    mindful_journaling_panel()
elif st.session_state["page"] == "Journal Analysis":
    journal_analysis_panel()
elif st.session_state["page"] == "Mindful Breathing":
    mindful_breathing_panel()
elif st.session_state["page"] == "IoT Dashboard (ECE)":
    iot_dashboard_panel()
elif st.session_state["page"] == "Wellness Check-in":
    wellness_checkin_panel()
elif st.session_state["page"] == "Report & Summary":
    report_summary_panel()