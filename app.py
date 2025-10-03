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
import numpy as np

# Lightweight sentiment analyzer cached
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import the OpenAI library (used for OpenRouter compatibility)
from openai import OpenAI
from openai import APIError

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

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="AI Wellness Companion", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- ECE HELPER FUNCTIONS ----------

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
    
    # Ensure current_time_ms is float for sin function
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
    # FIX: Ensure phq9_score is handled safely if None
    phq9_score = st.session_state.get("phq9_score") or 0
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
def setup_ai_model(api_key: str):
    """Lazy configure OpenAI client for OpenRouter."""
    if not api_key:
        return None, False, None
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
        
        history = [{"role": "system", "content": system_instruction}]

        if "chat_messages" in st.session_state:
            for msg in st.session_state["chat_messages"]:
                if msg["role"] in ["user", "assistant"]:
                     history.append(msg)
        
        if len(history) <= 1:
             history.append({"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"})

        return client, True, history
    except Exception:
        return None, False, None

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

# ---------- Session state defaults ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "kalman_state" not in st.session_state:
    st.session_state["kalman_state"] = initialize_kalman()
if "physiological_data" not in st.session_state:
    st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])

if "_ai_model" not in st.session_state:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY)
    st.session_state["_ai_model"] = _ai_client_obj 
    st.session_state["_ai_available"] = _ai_available
    st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]
    
if "_supabase_client_obj" not in st.session_state:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
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

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]


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

        if messages_for_api[-1]["content"] != prompt_clean:
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

def generate_wordcloud_figure_if_possible(text: str):
    if not text or not text.strip():
        return None
    try:
        from wordcloud import WordCloud
        stopwords = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'it', 'I', 'my', 'me', 'that', 'this', 'for', 'was', 'with'])
        wc = WordCloud(
            width=800, height=400, background_color="#f7f9fb", stopwords=stopwords, 
            max_words=100, contour_width=3, contour_color='#4a90e2'
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception:
        return None

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

# ---------- UI style (UPGRADED FOR PREMIUM LOOK) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* 1. Global Background and Typography */
    .stApp { 
        background: #ffffff; /* Clean white background */
        color: #2c3e50; 
        font-family: 'Poppins', sans-serif; 
    }
    .main .block-container { 
        padding: 2rem 5rem; /* Wider padding for better spacing */
    }
    
    /* 2. Sidebar Aesthetics */
    .css-1d3f90z { /* Streamlit sidebar selector */
        background-color: #f7f9fb; /* Light background for sidebar */
    }
    
    /* 3. Custom Card Style (Stronger shadow for depth) */
    .card { 
        background-color: #f7f9fb; /* Slightly off-white background */
        border-radius: 20px; /* More rounded corners */
        box-shadow: 0 10px 20px rgba(74, 144, 226, 0.15); /* Blue-tinted, stronger shadow */
        padding: 25px; 
        margin-bottom: 25px; 
        border: 1px solid #e0e8f0; /* Subtle border */
        transition: all .2s ease-in-out; 
    }
    .card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 15px 30px rgba(74, 144, 226, 0.25); 
    }

    /* 4. Primary Button Style (High-Impact Pill Shape) */
    .stButton>button { 
        color: #fff; 
        background: linear-gradient(135deg, #4a90e2 0%, #7dace8 100%); /* Subtle gradient */
        border-radius: 30px; /* Pill shape */
        padding: 10px 25px; 
        font-weight: 600; 
        border: none; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all .2s;
    }
    .stButton>button:hover { 
        opacity: 0.9; 
        transform: scale(1.02);
    }
    
    /* 5. Custom Sidebar Status */
    .sidebar-status {
        padding: 5px 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .status-connected { background-color: #e6ffee; color: #155724; border-left: 4px solid #28a745; }
    .status-local { background-color: #fff8e6; color: #856404; border-left: 4px solid #ffc107; }
    
    /* 6. Larger Titles for Impact */
    h1 { color: #2c3e50; font-weight: 700; margin-bottom: 0.5rem; }
    h2 { color: #34495e; font-weight: 600; }
    h3 { color: #597d9e; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation and Status
st.sidebar.markdown("### Status")

# Custom Status Blocks
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


# ---------- Sidebar: Auth (Retained) ----------
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
                st.session_state[key] = None
            st.session_state["daily_journal"] = []
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            st.session_state.chat_messages = []
            
            OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY)
            st.session_state["_ai_model"] = _ai_client_obj
            st.session_state["_ai_available"] = _ai_available
            st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]

            st.sidebar.info("Logged out.")
            st.rerun()

sidebar_auth()


# ---------- Panels (MODIFIED FOR UX) ----------
def homepage_panel():
    # Use HTML for the main title to ensure font is applied
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #4a90e2;'>🧠</span></h1>", unsafe_allow_html=True)
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    
    st.markdown("---")
    
    # --- Row 1: Daily Inspiration & Quick Actions ---
    col_main, col_img = st.columns([7, 3])
    
    with col_main:
        st.markdown("<h2>Daily Focus ✨</h2>", unsafe_allow_html=True)
        # Use a large, centered quote card
        st.markdown("<div class='card' style='text-align:center; border-left: 5px solid #2ecc71;'>", unsafe_allow_html=True)
        st.markdown(f"**<span style='font-size: 1.25rem;'>“{random.choice(QUOTES)}”</span>**", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h3>Quick Actions</h3>")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🧘‍♀️ Start Breathing", key="home_breath_btn", use_container_width=True):
                st.session_state["page"] = "Mindful Breathing"
                st.rerun()
        with c2:
            if st.button("💬 Talk to AI", key="home_chat_btn", use_container_width=True):
                st.session_state["page"] = "AI Chat"
                st.rerun()
        with c3:
            if st.button("📝 Journal Now", key="home_journal_btn", use_container_width=True):
                st.session_state["page"] = "Mindful Journaling"
                st.rerun()
                
    with col_img:
        st.image("https://images.unsplash.com/photo-1549490349-f06b3e942007?q=80&w=2070&auto=format&fit=crop", 
                 caption="Take a moment for yourself", 
                 use_column_width=True)

    st.markdown("---")
    
    # --- Row 2: Feature Cards ---
    st.markdown("<h2>Your Toolkit</h2>")
    
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("Mood Tracker", "📈 Log quick ratings, earn badges.", "#f39c12"),
        ("AI Chat", "💬 Empathetic AI companion, always ready.", "#4a90e2"),
        ("Journal & Insights", "📊 Track progress, visualize emotions.", "#2ecc71"),
        ("IoT Dashboard (ECE)", "⚙️ Real-time heart-rate/stress monitoring (DSP Demo).", "#e74c3c")
    ]
    
    for i, (title, desc, color) in enumerate(features):
        with [f1, f2, f3, f4][i]:
            st.markdown(f"""
            <div class='card' style='padding: 15px; border-left: 5px solid {color};'>
                <h4 style='margin-top:0;'>{title}</h4>
                <p style='font-size: 0.9rem; margin-bottom: 0;'>{desc}</p>
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

                # Streak Logic (retained)
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

                # Badge check (retained)
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

    # Use a container with fixed height and scrolling for a chat-app feel
    with st.container(height=500):
        for message in st.session_state.chat_messages:
            if message["role"] in ["user", "assistant"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    prompt = st.chat_input("What's on your mind?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Listening closely..."):
                ai_response = safe_generate(prompt)
                st.markdown(ai_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                        
        st.rerun()

def mindful_breathing_panel():
    st.header("Mindful Breathing 🧘‍♀️")
    st.markdown("Follow the prompts for 3 cycles: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Breathe deep and recenter.")
    
    if "breathing_state" not in st.session_state:
        st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}

    bs = st.session_state["breathing_state"]
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("Start 3 Cycles", key="start_breathing_btn", use_container_width=True)
    with col_btn2:
        if st.button("Reset", key="reset_breathing_btn", use_container_width=True) and not bs["running"]:
            st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}
            st.rerun()
            return

    if start_btn and not bs["running"]:
        bs["running"] = True
        bs["start_time"] = time.time()
        bs["cycles_done"] = 0
        st.session_state["breathing_state"] = bs
        st.rerun()

    if bs["running"]:
        PHASES = [("Inhale 🌬️ (Expand)", 4.0, "#4a90e2"), ("Hold ⏸️ (Pause)", 4.0, "#357bd9"), ("Exhale 💨 (Release)", 6.0, "#f39c12")]
        total_cycle_time = sum(p[1] for p in PHASES)
        elapsed = time.time() - (bs["start_time"] or time.time())
        
        cycle_number = int(elapsed // total_cycle_time) + 1
        time_in_cycle = elapsed % total_cycle_time

        if cycle_number > 3:
            bs["running"] = False
            bs["cycles_done"] = 3
            st.session_state["breathing_state"] = bs
            st.success("Exercise complete! You did a great job resetting your mind. 🌟")
            
            if "Breathing Master" not in st.session_state["streaks"]["badges"]:
                st.session_state["streaks"]["badges"].append("Breathing Master")
                
            st.rerun()
            return

        st.info(f"Cycle {cycle_number} of 3")
        
        phase_start = 0.0
        current_phase_color = ""
        
        for phase, duration, color in PHASES:
            if time_in_cycle < phase_start + duration:
                time_in_phase = time_in_cycle - phase_start
                progress = min(max(time_in_phase / duration, 0.0), 1.0)
                time_remaining = duration - time_in_phase
                
                current_phase_color = color
                
                # Use HTML to center and style the text for a nicer look
                st.markdown(f"<h2 style='text-align:center;color:{current_phase_color};font-size: 2.5rem;'>{phase}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align:center;color:{current_phase_color};font-size: 2rem;'>{time_remaining:.1f}s</h3>", unsafe_allow_html=True)

                st.progress(progress)
                break
            phase_start += duration
        
        time.sleep(0.1)
        st.rerun()

def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    st.markdown("Write freely about your day, your feelings, or anything on your mind. Your words are private.")
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        journal_text = st.text_area("Today's reflection", height=220, key="journal_text")
        
        col_save, col_info = st.columns([1,2])
        with col_save:
            if st.button("Save Entry", key="save_entry_btn", use_container_width=True):
                if journal_text.strip():
                    sent = sentiment_compound(journal_text)
                    
                    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry_data = {"date": date_str, "text": journal_text, "sentiment": sent}
                    
                    if st.session_state.get("logged_in") and st.session_state.get("_db_connected") and st.session_state.get("user_id"):
                        ok = save_journal_db(st.session_state.get("user_id"), journal_text, sent)
                        if ok:
                            st.success("Saved to your account on Supabase. Your data is secure! 🔒")
                        else:
                            st.warning("Could not save to DB. Saved locally for now.")
                            st.session_state["daily_journal"].append(entry_data)
                    else:
                        st.session_state["daily_journal"].append(entry_data)
                        st.success("Saved locally to this browser session. Log in to save permanently. 💾")
                        
                    st.session_state["journal_text"] = "" 
                    st.rerun()
                else:
                    st.warning("Write something you want to save.")
        
        with col_info:
            st.info("Your data is analyzed privately for trends.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Recent Entries")
    
    recent_cols = st.columns(3)
    
    if st.session_state["daily_journal"]:
        for i, entry in enumerate(reversed(st.session_state["daily_journal"][-6:])): # Show last 6 in 3 columns
            date = pd.to_datetime(entry['date']).strftime('%Y-%m-%d')
            sentiment = entry.get('sentiment', 0)
            
            if sentiment >= 0.05:
                label = "🟢 Positive"
                color = "#2ecc71"
            elif sentiment <= -0.05:
                label = "🔴 Negative"
                color = "#e74c3c"
            else:
                label = "⚫ Neutral"
                color = "#95a5a6"
            
            with recent_cols[i % 3]:
                 st.markdown(f"<div class='card' style='padding:12px; border-left: 5px solid {color}; margin-bottom: 15px;'>**{date}** ({label}) <br><small style='color: #7f8c8d;'>{entry.get('text')[:40]}...</small></div>", unsafe_allow_html=True, help=entry.get('text'))
    else:
        st.info("_No entries saved yet._")

def journal_analysis_panel():
    st.header("Journal & Analysis 📊")
    
    all_text = get_all_user_text()
    if not all_text:
        st.info("No journal or chat text yet — start journaling or talking to get insights.")
        return

    entries = []
    for e in st.session_state["daily_journal"]:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e.get("sentiment", 0), "source": "Journal"})

    if entries:
        df = pd.DataFrame(entries).sort_values("date")
        df["sentiment_label"] = df["compound"].apply(lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
        
        st.markdown("<h2>Emotional Trend Over Time</h2>", unsafe_allow_html=True)
        fig = px.line(df, x="date", y="compound", color="sentiment_label", markers=True,
                      title="Sentiment Score Trend (VADER)",
                      color_discrete_map={"Positive":"#2ecc71","Neutral":"#95a5a6","Negative":"#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

        col_cloud, col_dist = st.columns(2)
        
        with col_cloud:
            st.markdown("<h2>Top Words & Focus ☁️</h2>", unsafe_allow_html=True)
            wc_fig = generate_wordcloud_figure_if_possible(all_text)
            if wc_fig:
                 st.pyplot(wc_fig)
            else:
                 st.info("Could not generate word cloud (missing package or too little data).")
                 
        with col_dist:
            st.markdown("<h2>Sentiment Distribution</h2>", unsafe_allow_html=True)
            fig_hist = px.histogram(df, x="compound", color="sentiment_label", 
                                    title="Frequency of Sentiment Scores", 
                                    color_discrete_map={"Positive":"#2ecc71","Neutral":"#95a5a6","Negative":"#e74c3c"})
            st.plotly_chart(fig_hist, use_container_width=True)

def iot_dashboard_panel():
    st.header("IoT Dashboard (ECE Focus) ⚙️")
    st.markdown("""
    <p style='background-color:#f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4a90e2;'>
    This panel simulates **real-time hardware data** (PPG/GSR sensor) and demonstrates **Digital Signal Processing (DSP)** using a **Kalman Filter** to clean the noisy heart rate signal—a key Computer Engineering skill.
    </p>
    """, unsafe_allow_html=True)

    if "stream_running" not in st.session_state:
        st.session_state["stream_running"] = False
        
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start Simulated Data Stream", key="start_stream_btn", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Stop Stream", key="stop_stream_btn", use_container_width=True)

    if start_btn:
        st.session_state["stream_running"] = True
        st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
        st.session_state["kalman_state"] = initialize_kalman()
        st.rerun()

    if stop_btn:
        st.session_state["stream_running"] = False
        st.rerun()
        
    # --- Real-Time Data Logic ---
    if st.session_state["stream_running"]:
        
        st.markdown("---")
        st.subheader("Real-Time Metrics")
        
        current_time_ms = int(time.time() * 1000)
        new_data_point = generate_simulated_physiological_data(current_time_ms)
        
        kalman_state = st.session_state["kalman_state"]
        filtered_hr_bpm, new_kalman_state = kalman_filter_simple(
            new_data_point["raw_ppg_signal"], kalman_state
        )
        st.session_state["kalman_state"] = new_kalman_state
        
        new_data_point["filtered_hr"] = filtered_hr_bpm
        
        df = pd.concat([st.session_state["physiological_data"], pd.DataFrame([new_data_point])], ignore_index=True)
        df = df.tail(150)
        st.session_state["physiological_data"] = df
        
        c1, c2, c3 = st.columns(3)
        with c1:
             c1.metric("Raw PPG Reading (V)", f"{new_data_point['raw_ppg_signal']:.2f} V", "Noisy Input")
        with c2:
             # Highlight DSP output
             c2.metric("Filtered HR (BPM)", f"{filtered_hr_bpm:.1f}", "DSP Output", delta_color="off")
        with c3:
             stress_label = "High" if new_data_point['gsr_stress_level'] > 1.5 else "Low/Normal"
             c3.metric("GSR Stress Level", f"{new_data_point['gsr_stress_level']:.2f}", stress_label)


        # --- Real-Time Scrolling Plot (Plotly) ---
        st.subheader("Signal Visualization (Raw vs. Filtered)")
        
        df_melt = df.melt(id_vars='time_ms', value_vars=['raw_ppg_signal', 'filtered_hr'], 
                          var_name='Signal Type', value_name='Value')

        fig = px.line(df_melt, x='time_ms', y='Value', color='Signal Type', 
                      title='Kalman Filter DSP Demonstration',
                      color_discrete_map={'raw_ppg_signal': '#e74c3c', 'filtered_hr': '#2ecc71'})
        
        fig.update_traces(line=dict(shape='spline', width=2))
        fig.update_xaxes(visible=False, title="") 
        
        placeholder = st.empty()
        placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.5)
        st.rerun()
        
    else:
        st.info("Stream is currently stopped. Press 'Start Simulated Data Stream' to begin the ECE demonstration.")

def wellness_checkin_panel():
    st.header("Wellness Check-in (PHQ-9) 🩺")
    st.markdown("A brief screening tool to help you reflect on your mental well-being. Your answers are private.")

    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself—or that you are a failure or have let yourself or your family down?"
    ]

    score_map = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
    
    responses = {}
    st.markdown("<div class='card' style='border-left: 5px solid #f39c12;'>", unsafe_allow_html=True)
    for i, q in enumerate(questions):
        responses[i] = st.radio(f"**{i+1}.** {q}", options=list(score_map.keys()), key=f"phq9_q{i}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Calculate Score", key="calculate_phq9", use_container_width=True):
        total_score = sum(score_map[r] for r in responses.values())
        
        if total_score <= 4:
            interpretation = "Minimal to None. You seem well! Keep nurturing your mental health. 🌱"
        elif total_score <= 9:
            interpretation = "Mild. Be mindful of these feelings and consider reaching out."
        elif total_score <= 14:
            interpretation = "Moderate. Please consider reaching out to a mental health professional or a trusted adult."
        else:
            interpretation = "Moderately Severe to Severe. This requires attention. **Please contact a professional immediately.** You deserve support."

        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation
        st.rerun()

    if st.session_state["phq9_score"] is not None:
        st.markdown("---")
        st.subheader("Your Result")
        st.metric(label="Total Score (Max 18)", value=f"{st.session_state['phq9_score']}", delta_color="off")
        st.markdown(f"<p style='padding: 15px; border-radius: 10px; background-color: #eaf0ff; border-left: 5px solid #4a90e2;'>**Interpretation:** {st.session_state['phq9_interpretation']}</p>", unsafe_allow_html=True)

def report_summary_panel():
    st.header("Comprehensive Wellness Report 📄")
    st.markdown("A snapshot of your long-term progress across all app features.")
    
    # --- MOOD SUMMARY ---
    with st.container():
        st.markdown("<h2>1. Mood Metrics</h2>", unsafe_allow_html=True)
        if st.session_state["mood_history"]:
            df_mood = pd.DataFrame(st.session_state["mood_history"])
            avg_mood = df_mood['mood'].mean()
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Average Mood Score", f"{avg_mood:.1f}/11")
            col_m2.metric("Total Mood Logs", f"{len(df_mood)} days")
            col_m3.metric("Current Streak", f"{st.session_state['streaks'].get('mood_log',0)} days 🔥")
            
            fig_mood = px.bar(df_mood.groupby(pd.to_datetime(df_mood['date']).dt.date)['mood'].mean().reset_index(),
                              x='date', y='mood', title='Average Daily Mood')
            st.plotly_chart(fig_mood, use_container_width=True)
        else:
            st.info("No mood data yet.")
        st.markdown("---")

    # --- JOURNAL SUMMARY ---
    with st.container():
        st.markdown("<h2>2. Journal Analysis</h2>", unsafe_allow_html=True)
        if st.session_state["daily_journal"]:
            df_journal = pd.DataFrame(st.session_state["daily_journal"])
            avg_sent = df_journal['sentiment'].mean()
            
            col_j1, col_j2 = st.columns(2)
            col_j1.metric("Average Sentiment", f"{avg_sent:.2f}")
            col_j2.metric("Total Entries", f"{len(df_journal)} entries")
            
            fig_sent = px.histogram(df_journal, x="sentiment", nbins=20, title="Journal Sentiment Distribution")
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("No journal data yet.")
        st.markdown("---")

    # --- CHECK-IN SUMMARY ---
    with st.container():
        st.markdown("<h2>3. Recent Check-in & Badges</h2>", unsafe_allow_html=True)
        col_check, col_badge = st.columns(2)
        
        with col_check:
             if st.session_state["phq9_score"] is not None:
                st.markdown(f"**Last Check-in Score:** `{st.session_state['phq9_score']}`")
                st.info(f"**Interpretation:** {st.session_state['phq9_interpretation']}")
             else:
                 st.info("No recent check-in score.")

        with col_badge:
            st.subheader("Badges Earned")
            if st.session_state["streaks"]["badges"]:
                st.markdown(", ".join([f"**{b}** 🌟" for b in st.session_state["streaks"]["badges"]]))
            else:
                st.info("No badges earned yet. Complete more activities!")


# ---------- Main App Dispatcher ----------
page_dispatch = {
    "Home": homepage_panel,
    "AI Chat": ai_chat_panel,
    "Mood Tracker": mood_tracker_panel,
    "Mindful Journaling": mindful_journaling_panel,
    "Journal Analysis": journal_analysis_panel,
    "Mindful Breathing": mindful_breathing_panel,
    "IoT Dashboard (ECE)": iot_dashboard_panel, 
    "Wellness Check-in": wellness_checkin_panel,
    "Report & Summary": report_summary_panel,
}

# Run the selected page function
page_dispatch.get(st.session_state["page"], homepage_panel)()