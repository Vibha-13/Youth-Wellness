import streamlit as st
import os
import time
import random
import io
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np

# Lightweight sentiment analyzer (KEEP THIS - it's simple and stable)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import the OpenAI library (used for OpenRouter compatibility)
from openai import OpenAI
from openai import APIError

# TEMPORARILY COMMENT OUT COMPLEX IMPORTS TO AVOID SPAWN ERROR
# from wordcloud import WordCloud # Still commented out, as it can cause issues.

# ---------- CONSTANTS ----------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" # You can change this to any OpenRouter model
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

PHQ9_INTERPRETATION = {
    (0, 4): "Minimal to None",
    (5, 9): "Mild",
    (10, 14): "Moderate",
    (15, 19): "Moderately Severe",
    (20, 27): "Severe"
}

# ---------- Streamlit page config ----------
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

# Call the setup function early in the main script flowsetup_page_and_layout()

# ---------- ECE HELPER FUNCTIONS (KEEP THESE) ----------

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
        # Import moved inside function for lazy loading
        from supabase import create_client
        client = create_client(url, key)
        return client, True
    except Exception:
        return None, False

# ---------- Session state defaults (CLEANED UP) ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "kalman_state" not in st.session_state:
    st.session_state["kalman_state"] = initialize_kalman()
if "physiological_data" not in st.session_state:
    st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])

if "_ai_model" not in st.session_state:
    # --- ROBUST SECRET LOADING ---
    raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
    
    # Initialize chat_messages safely
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    
    _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY, st.session_state["chat_messages"])
    st.session_state["_ai_model"] = _ai_client_obj 
    st.session_state["_ai_available"] = _ai_available
    st.session_state["chat_messages"] = _chat_history_list
    
if "_supabase_client_obj" not in st.session_state:
    # --- ROBUST SECRET LOADING ---
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

if "phq9_score" not in st.session_state:
    st.session_state["phq9_score"] = None

if "phq9_interpretation" not in st.session_state:
    st.session_state["phq9_interpretation"] = None

if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop" # 'stop', 'running', 'finished'

analyzer = setup_analyzer()

# ---------- Helper functions (Retained) ----------
def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300):
    """
    Generate text via OpenRouter, using the system message and current history.
    """
    prompt_lower = prompt.lower()
    
    # Custom, empathetic responses for key phrases
    if any(phrase in prompt_lower for phrase in ["demotivated", "heavy", "don't want to do anything", "feeling down"]):
        return (
            "Thanks for reaching out and sharing that with me. Honestly, **that feeling of demotivation can be really heavy, and it takes a lot of courage just to name it.** I want you to know you're definitely not alone in feeling this way. Before we try to tackle the whole mountain, let's just look at one rock. **Is there one tiny task or thought that feels the heaviest right now?** Sometimes just describing it makes it a little lighter. 🌱"
        )
    elif "funny" in prompt_lower or "joke" in prompt_lower or "break" in prompt_lower:
        previous_topic = "our chat"
        user_messages = [m for m in st.session_state.chat_messages if m["role"] == "user"]
        if len(user_messages) > 1:
            previous_prompt = user_messages[-2]["content"]
            previous_topic = f"what you were sharing about '{previous_prompt[:25]}...'"

        return (
            "I hear you! It sounds like you need a quick reset, and a little humor is a great way to do that. **Okay, here's a silly one that always makes me smile:** Why don't scientists trust atoms? **Because they make up everything!** 😂 I hope that got a small chuckle! **Ready to dive back into** " + previous_topic + ", **or should I keep the jokes coming for a few more minutes?**"
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
            if st.session_state.get("_ai_available"):
                 st.error("OpenRouter API Error. Please check your key or try a different model.")
            pass
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
        # Simplified user registration for prototype
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

# ---------- UI style (RETAINED) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* 1. Global Background and Typography */
    .stApp { 
        background: #F0F2F6; /* Very Light Grey/Off-White background */
        color: #1E1E1E; 
        font-family: 'Poppins', sans-serif; 
    }
    .main .block-container { 
        padding: 2rem 4rem; /* Generous padding */
    }
    
    /* 2. Sidebar Aesthetics */
    .css-1d3f90z { /* Streamlit sidebar selector */
        background-color: #FFFFFF; /* White background for sidebar */
        box-shadow: 2px 0 10px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    
    /* 3. Custom Card Style (The Core Mobile App Look) */
    .card { 
        background-color: #FFFFFF; /* White background for the card */
        border-radius: 16px; /* Highly rounded corners */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); /* Soft, noticeable shadow for lift */
        padding: 25px; /* Slightly more padding */
        margin-bottom: 20px; 
        border: none; /* Removed subtle border for cleaner lift effect */
        transition: all .2s ease-in-out; 
    }
    .card:hover { 
        transform: translateY(-3px); /* Subtle hover lift */
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15); 
    }

    /* 4. Primary Button Style (Vibrant and Rounded) */
    .stButton>button { 
        color: #FFFFFF; 
        background: #5D54A4; /* Deep Purple (similar to the MindHealth app) */
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
    
    /* 5. Custom Sidebar Status */
    .sidebar-status {
        padding: 5px 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    /* Updated colors to be vibrant on a white sidebar */
    .status-connected { background-color: #D4EDDA; color: #155724; border-left: 4px solid #28A745; }
    .status-local { background-color: #FFEEDD; color: #856404; border-left: 4px solid #FFC107; }
    
    /* 6. Larger Titles for Impact */
    h1 { color: #1E1E1E; font-weight: 700; margin-bottom: 0.5rem; }
    h2 { color: #333333; font-weight: 600; margin-top: 2rem;}
    h3 { color: #5D54A4; font-weight: 500; margin-top: 1rem;}

    /* Breathing Circle Styles */
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

# Sidebar Navigation and Status (RETAINED)
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


# ---------- Sidebar: Auth (RETAINED) ----------
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
                        st.sidebar.success("Logged in and data loaded.")
                    elif st.session_state.get("_db_connected") is False:
                         st.sidebar.info("Logged in locally (no DB).")
                         
                    st.rerun()

                else:
                    uid = register_user_db(email)
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.sidebar.success("Registered & logged in.")
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
            
            st.session_state["daily_journal"] = []
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            
            # Reset AI setup logic
            raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
            _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY, [])
            st.session_state["_ai_model"] = _ai_client_obj
            st.session_state["_ai_available"] = _ai_available
            st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]

            st.sidebar.info("Logged out.")
            st.rerun()

sidebar_auth()


# ---------- Panels (MODIFIED FOR UX) ----------
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
    
    # --- Row 2: Quick Actions (Similar to the mobile app's icon buttons) ---
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
    
    # --- Row 3: Feature Cards (Mimicking the spaced-out, illustrated cards) ---
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
                    except Exception:
                        continue
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")

    col_stats, col_badges = st.columns([2,1])

    with col_stats:
        # Plot mood history (UPGRADED TO PLOTLY)
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


def ai_chat_panel():
    st.header("AI Chat 💬")
    st.markdown("Your compassionate AI buddy. Start chatting, and I’ll listen.")

    with st.container(height=500, border=True):
        for message in st.session_state.chat_messages:
            if message["role"] in ["user", "assistant"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    prompt = st.chat_input("What's on your mind?")
    if prompt:
        
        # Check if the last message was already this prompt (to prevent double-append on rerun)
        if not st.session_state.chat_messages or st.session_state.chat_messages[-1]["content"] != prompt or st.session_state.chat_messages[-1]["role"] != "user":
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Listening closely..."):
                ai_response = safe_generate(prompt)
                st.markdown(ai_response)
                
                # Check if the last message was already this response (to prevent double-append on rerun)
                if not st.session_state.chat_messages or st.session_state.chat_messages[-1]["content"] != ai_response or st.session_state.chat_messages[-1]["role"] != "assistant":
                    st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                        
        st.rerun()

def mindful_breathing_panel():
    st.header("Mindful Breathing 🧘‍♀️")
    st.markdown("Follow the prompts for 3 cycles: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Breathe deep and recenter.")
    
    
    if st.session_state["breathing_state"] == "stop":
        if st.button("Start 3-Cycle Breath", key="start_breathing_btn", use_container_width=True):
            st.session_state["breathing_state"] = "running"
            st.rerun()
        st.markdown("<div class='card' style='text-align:center;'>Tap start to begin the exercise.</div>", unsafe_allow_html=True)

    if st.session_state["breathing_state"] == "running":
        
        placeholder = st.empty()
        
        # Total time is 3 cycles * (4 + 4 + 6) = 42 seconds
        # Use st.progress to track the full progress
        full_progress_bar = st.progress(0, text="0 / 42 seconds complete...")
        
        for cycle in range(1, 4):
            # Inhale (4s)
            with placeholder.container():
                st.markdown(f"<h2>Cycle {cycle}/3: <span style='color:#28A745;'>INHALE DEEPLY</span></h2>", unsafe_allow_html=True)
                st.markdown("<div class='breathing-circle breathe-inhale' style='background-color:#28A745;'>INHALE</div>", unsafe_allow_html=True)
            time.sleep(4)
            full_progress_bar.progress(int((cycle-1) * 14 + 4) / 42.0, text=f"{(cycle-1) * 14 + 4} / 42 seconds complete... (Holding)")

            # Hold (4s)
            with placeholder.container():
                st.markdown(f"<h2>Cycle {cycle}/3: <span style='color:#FFC107;'>HOLD</span></h2>", unsafe_allow_html=True)
                st.markdown("<div class='breathing-circle' style='background-color:#FFC107;'>HOLD</div>", unsafe_allow_html=True)
            time.sleep(4)
            full_progress_bar.progress(int((cycle-1) * 14 + 8) / 42.0, text=f"{(cycle-1) * 14 + 8} / 42 seconds complete... (Exhaling)")

            # Exhale (6s)
            with placeholder.container():
                st.markdown(f"<h2>Cycle {cycle}/3: <span style='color:#FF5733;'>EXHALE SLOWLY</span></h2>", unsafe_allow_html=True)
                st.markdown("<div class='breathing-circle breathe-exhale' style='background-color:#FF5733;'>EXHALE</div>", unsafe_allow_html=True)
            time.sleep(6)
            full_progress_bar.progress(int(cycle * 14) / 42.0, text=f"{cycle * 14} / 42 seconds complete...")


        st.session_state["breathing_state"] = "finished"
        placeholder.empty()
        st.rerun()

    if st.session_state["breathing_state"] == "finished":
        st.success("✅ **Breathing exercise complete!** You finished 3 cycles and took a moment for yourself.")
        if st.button("Start Again", key="restart_breathing_btn", use_container_width=True):
            st.session_state["breathing_state"] = "stop"
            st.rerun()
        if st.button("Go Home", key="finish_breathing_btn", use_container_width=True):
            st.session_state["page"] = "Home"
            st.session_state["breathing_state"] = "stop"
            st.rerun()

def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    st.markdown("Use this private space to reflect on your day, thoughts, and emotions.")
    
    # --- TEMPORARY CSS FIX: FORCE TEXT COLOR TO BLACK ---
    # This aggressively ensures your input text is visible if it was set to white by default.
    st.markdown("""
        <style>
        textarea {
            color: black !important; /* Force text color to black */
        }
        </style>
    """, unsafe_allow_html=True)
    # ----------------------------------------------------
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # --- SIMPLE KEY RETAINED ---
        # We will clear this exact key later.
        ENTRY_KEY = "journal_input_final" 
        
        entry_text = st.text_area(
            "What's on your mind right now?", 
            height=250, 
            key=ENTRY_KEY, # Use the defined key
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
                        st.session_state["page"] = "Journal Analysis" 
                        st.rerun()
                
                # --- CRITICAL FIX: CLEAR THE EXACT MATCHING KEY ---
                st.session_state[ENTRY_KEY] = ""
                # ------------------------------------------------
                
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

def journal_analysis_panel():
    st.header("Journal Analysis 📊")
    st.markdown("See the data behind your words. This analysis covers all journal and chat entries.")
    
    all_text = get_all_user_text()
    
    # --- Show Reframing Card if just generated ---
    if st.session_state.get("last_reframing_card"):
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #5D54A4;'>
            <h3 style='margin-top:0;'>Your Coping Card (Reframing) 💡</h3>
            <p style='font-size: 1.1rem; font-style: italic; color: #1E1E1E;'>"{st.session_state["last_reframing_card"]}"</p>
        </div>
        """, unsafe_allow_html=True)
        # Clear after showing once
        st.session_state["last_reframing_card"] = None 
        st.markdown("---")


    if not st.session_state["daily_journal"] and not all_text:
        st.warning("Start journaling or chatting with the AI to generate analysis data.")
        return

    # 1. Overall Sentiment Metric
    avg_journal_sentiment = 0.0
    if st.session_state["daily_journal"]:
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        avg_journal_sentiment = df_journal["sentiment"].mean()
    
    col_metric, col_word_count = st.columns(2)
    
    with col_metric:
        sentiment_label = "Positive" if avg_journal_sentiment > 0.1 else ("Negative" if avg_journal_sentiment < -0.1 else "Neutral")
        sentiment_delta = f"{avg_journal_sentiment:.2f}"
        st.metric(
            label="Average Sentiment Score (Journal)", 
            value=sentiment_label, 
            delta=sentiment_delta,
            delta_color="normal" if avg_journal_sentiment > 0 else "inverse"
        )
        st.markdown(f"**Total Entries:** {len(st.session_state['daily_journal'])}")

    with col_word_count:
        st.metric("Total Words Analyzed", f"{len(all_text.split()):,}")

    st.markdown("---")

    # 2. Sentiment History Plot
    st.subheader("Sentiment Trend")
    if st.session_state["daily_journal"]:
        df_journal_plot = pd.DataFrame(st.session_state["daily_journal"]).copy()
        df_journal_plot['date'] = pd.to_datetime(df_journal_plot['date'])
        
        fig = px.bar(
            df_journal_plot, 
            x='date', 
            y='sentiment', 
            title='Journal Sentiment Over Time',
            color='sentiment',
            color_continuous_scale=px.colors.diverging.RdYlGn, # Red-Yellow-Green scale
            template="plotly_white"
        )
        fig.update_layout(yaxis_range=[-1, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Write a few journal entries to see your sentiment history!")


    # 3. Word Cloud (Still commented out to ensure stability)
    # fig_wc = generate_wordcloud_figure_if_possible(all_text)
    # if fig_wc:
    #     st.subheader("Most Used Words (Word Cloud)")
    #     st.pyplot(fig_wc)
    # else:
    #     st.info("Word Cloud feature is currently disabled to ensure stable deployment.")
        
    
def iot_dashboard_panel():
    st.header("IoT Dashboard (ECE Demo) ⚙️")
    st.markdown("Real-time simulated physiological data monitoring and Kalman filter demonstration.")
    
    # ------------------
    # Data Simulation Logic
    # ------------------
    current_time_ms = int(time.time() * 1000)
    data = generate_simulated_physiological_data(current_time_ms)
    
    # 1. Kalman Filter Application (HR)
    # The raw PPG signal is the measurement (z_meas)
    filtered_hr, st.session_state["kalman_state"] = kalman_filter_simple(
        data["raw_ppg_signal"], 
        st.session_state["kalman_state"]
    )
    data["filtered_hr"] = filtered_hr
    
    # Append new data to the session state DataFrame
    new_row = pd.DataFrame([data])
    st.session_state["physiological_data"] = pd.concat([st.session_state["physiological_data"], new_row], ignore_index=True)
    
    # Keep the DataFrame size manageable
    max_rows = 150
    if len(st.session_state["physiological_data"]) > max_rows:
        st.session_state["physiological_data"] = st.session_state["physiological_data"].iloc[-max_rows:]
        
    # ------------------
    # Dashboard Display
    # ------------------
    
    col_hr, col_gsr = st.columns(2)
    
    with col_hr:
        st.markdown(f"<h2>Filtered Heart Rate (BPM)</h2>", unsafe_allow_html=True)
        st.metric(label="Filtered HR", value=f"{filtered_hr:.1f}", delta_color="off")
    
    with col_gsr:
        st.markdown(f"<h2>GSR Stress Level</h2>", unsafe_allow_html=True)
        st.metric(label="GSR Stress", value=f"{data['gsr_stress_level']:.2f}", delta_color="off")
    
    st.markdown("---")
    
    st.subheader("Heart Rate: Raw vs. Kalman Filtered")
    
    # Prepare data for plotting
    df_plot = st.session_state["physiological_data"][["time_ms", "raw_ppg_signal", "filtered_hr"]].melt(
        id_vars=['time_ms'], var_name='variable', value_name='BPM'
    )
    
    fig = px.line(df_plot, x='time_ms', y='BPM', color='variable',
                  line_dash='variable',
                  color_discrete_map={'raw_ppg_signal': '#FF5733', 'filtered_hr': '#5D54A4'},
                  labels={'time_ms': 'Time', 'BPM': 'BPM'})
    fig.update_layout(legend_title_text='Signal Type')
    st.plotly_chart(fig, use_container_width=True)
    
    # Automatic refresh
    time.sleep(1)
    st.rerun()


def wellness_checkin_panel():
    st.header("Wellness Check-in 🩺")
    st.markdown("This section helps screen for symptoms of depression over the **last two weeks**. This is a simplified version of the PHQ-9 (Patient Health Questionnaire-9).")
    
    # Check if already completed today
    today_str = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.get("last_checkin_date") == today_str:
        st.success(f"You've already completed today's check-in (Score: **{st.session_state['phq9_score']}** - **{st.session_state['phq9_interpretation']}**). Come back tomorrow!")
        st.markdown("---")
        st.subheader("Your Last Check-in Result")
        st.markdown(f"**Score:** `{st.session_state['phq9_score']}`")
        st.markdown(f"**Interpretation:** **{st.session_state['phq9_interpretation']}**")
        return

    
    # PHQ-9 Questions Setup
    scores = {}
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Using columns for a cleaner layout
    col_q, col_s = st.columns([4, 1])
    col_q.markdown("**Symptom**")
    col_s.markdown("**Score**")
    st.markdown("---")
    
    # Question loop
    for i, question in enumerate(PHQ9_QUESTIONS):
        col_q, col_s = st.columns([4, 1])
        with col_q:
            st.markdown(question)
        with col_s:
            # Score options: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
            scores[i] = st.selectbox(
                label=f"q{i}", 
                options=[0, 1, 2, 3], 
                format_func=lambda x: f"{x}",
                key=f"phq9_q_{i}",
                label_visibility="collapsed"
            )
            
    st.markdown("</div>", unsafe_allow_html=True)

    
    if st.button("Calculate Check-in Score", key="calculate_phq9_btn", use_container_width=True):
        total_score = sum(scores.values())
        
        interpretation = "Interpretation unavailable"
        for (low, high), label in PHQ9_INTERPRETATION.items():
            if low <= total_score <= high:
                interpretation = label
                break
        
        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation
        st.session_state["last_checkin_date"] = today_str
        
        # Display feedback
        st.success(f"Check-in Complete! Your score is **{total_score}** ({interpretation}).")
        st.info("Remember, this is not a clinical diagnosis. Please reach out to a professional if you need support.")
        
        # Optionally, prompt AI for next steps based on high score
        if total_score >= 10 and st.session_state.get("_ai_available"):
            ai_prompt = f"The user just completed a check-in with a score of {total_score} ({interpretation}). Offer a very gentle, encouraging, and supportive message suggesting they chat with you or try a simple coping exercise (like breathing or a small positive action)."
            ref_thought = safe_generate(ai_prompt, max_tokens=150)
            st.session_state["last_reframing_card"] = ref_thought # Re-use the card mechanism for display
            
        st.rerun()


def report_summary_panel():
    st.header("Report & Summary 📄")
    st.markdown("A consolidated view of your well-being progress.")
    
    if not st.session_state["logged_in"]:
        st.warning("Please log in to view your personalized summary.")
        return
    
    col_metrics, col_stats = st.columns(2)
    
    # --- Metrics Column ---
    with col_metrics:
        st.subheader("Key Wellness Metrics")
        
        # Mood Metric
        avg_mood = 0
        if st.session_state["mood_history"]:
            df_mood = pd.DataFrame(st.session_state["mood_history"])
            avg_mood = df_mood["mood"].mean()
            
        st.metric("Avg. Mood Score (1-11)", f"{avg_mood:.1f}", delta_color="off")
        
        # PHQ-9 Metric
        phq9_score = st.session_state.get("phq9_score")
        if phq9_score is not None:
            st.metric("Last Check-in Score (PHQ-9)", f"{phq9_score} / 27", delta=st.session_state.get("phq9_interpretation"), delta_color="off")
        else:
            st.info("Complete a Wellness Check-in to see your score here.")

        # Journal Metric
        avg_sentiment = 0.0
        if st.session_state["daily_journal"]:
            df_journal = pd.DataFrame(st.session_state["daily_journal"])
            avg_sentiment = df_journal["sentiment"].mean()

        sentiment_label = "Positive" if avg_sentiment > 0.1 else ("Negative" if avg_sentiment < -0.1 else "Neutral")
        st.metric("Avg. Journal Sentiment", sentiment_label, delta=f"{avg_sentiment:.2f}")

    # --- Stats Column ---
    with col_stats:
        st.subheader("Progress & Badges")
        st.metric("Current Mood Streak", f"{st.session_state['streaks'].get('mood_log', 0)} Days")
        st.metric("Total Journal Entries", len(st.session_state["daily_journal"]))
        st.metric("Total Badges Earned", len(st.session_state["streaks"]["badges"]))
        
        if st.session_state["streaks"]["badges"]:
            st.markdown("---")
            st.markdown("**Your Badges:** " + " | ".join([f"**{b}** 🌟" for b in st.session_state["streaks"]["badges"]]))
            
    st.markdown("---")
    
    # --- Narrative Summary (AI Driven) ---
    st.subheader("Personalized Narrative")
    
    # Final AI Call for a comprehensive summary
    if st.session_state.get("_ai_available") and st.button("Generate Full AI Summary"):
        with st.spinner("Analyzing your data and writing your report..."):
            summary_prompt = f"""
            Generate a personalized, empathetic summary for the user based on the following data points:
            - Average Mood Score (1-11): {avg_mood:.1f}
            - Last PHQ-9 Score (0-27): {phq9_score or 'N/A'}
            - Average Journal Sentiment Score (-1 to 1): {avg_sentiment:.2f}
            - Total Journal Entries: {len(st.session_state['daily_journal'])}
            - Current Mood Streak: {st.session_state['streaks'].get('mood_log', 0)} days

            Write a supportive message that:
            1. Validates their journey and any consistency shown (streak/entries).
            2. Gently addresses the overall sentiment/mood score (if low, suggest focusing on small wins; if high, encourage maintenance).
            3. Provides one actionable, non-clinical piece of advice related to the data.
            Keep it under 150 words.
            """
            
            summary_response = safe_generate(summary_prompt, max_tokens=250)
            st.markdown(f"<div class='card'><p>{summary_response}</p></div>", unsafe_allow_html=True)
    else:
        st.info("Tap 'Generate Full AI Summary' to get a narrative report.")

# ---------- Page Renderer ----------
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