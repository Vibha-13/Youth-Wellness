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

# IMPORTANT: TEMPORARILY COMMENT OUT COMPLEX IMPORTS TO AVOID SPAWN ERROR
# import matplotlib.pyplot as plt # Keeping this as it might be needed by plotly/pandas, but be aware.
# from wordcloud import WordCloud # WordCloud is often an issue, so we'll skip it for now.
# Any specific audio libraries you may have imported (e.g., pyaudio, sounddevice) should be commented out here.

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

# ---------- Session state defaults (WITH ROBUST SECRET LOADING FIX) ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "kalman_state" not in st.session_state:
    st.session_state["kalman_state"] = initialize_kalman()
if "physiological_data" not in st.session_state:
    st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])

if "_ai_model" not in st.session_state:
    # --- UPGRADED ROBUST SECRET LOADING ---
    raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    # This line safely cleans the key regardless of quotes or format.
    OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
    
    _ai_client_obj, _ai_available, _chat_history_list = setup_ai_model(OPENROUTER_API_KEY)
    st.session_state["_ai_model"] = _ai_client_obj 
    st.session_state["_ai_available"] = _ai_available
    st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]
    
if "_supabase_client_obj" not in st.session_state:
    # --- UPGRADED ROBUST SECRET LOADING ---
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
    (Function body remains the same)
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
            # Re-enable error message if running on cloud and API key is set
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

# IMPORTANT: TEMPORARILY COMMENTING OUT WORDCLOUD FUNCTION TO AVOID SPAWN ERROR
def generate_wordcloud_figure_if_possible(text: str):
    """Placeholder for the WordCloud function to prevent import issues on deploy."""
    return None
    # if not text or not text.strip():
    #     return None
    # try:
    #     from wordcloud import WordCloud, STOPWORDS # Need this import if uncommenting
    #     stopwords = set(STOPWORDS)
    #     stopwords.update(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'it', 'I', 'my', 'me', 'that', 'this', 'for', 'was', 'with'])
    #     wc = WordCloud(
    #         width=800, height=400, background_color="#f7f9fb", stopwords=stopwords, 
    #         max_words=100, contour_width=3, contour_color='#4a90e2'
    #     ).generate(text)
    #     import matplotlib.pyplot as plt # Need this import if uncommenting
    #     fig, ax = plt.subplots(figsize=(8,4))
    #     ax.imshow(wc, interpolation="bilinear")
    #     ax.axis("off")
    #     return fig
    # except Exception:
    #     return None


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

# ---------- UI style (UPGRADED FOR MODERN LIGHT CARD LOOK) ----------
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
    # ... (function body remains the same)
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
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #5D54A4;'>🧠</span></h1>", unsafe_allow_html=True)
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    
    st.markdown("---")
    
    # --- Row 1: Daily Inspiration Card ---
    
    # Daily Inspiration Card
    with st.container():
        # Use a distinct color for the accent border
        st.markdown("<div class='card' style='border-left: 8px solid #FFC107;'>", unsafe_allow_html=True)
        st.markdown("<h3>Daily Inspiration ✨</h3>")
        st.markdown(f"**<span style='font-size: 1.25rem; font-style: italic;'>“{random.choice(QUOTES)}”</span>**", unsafe_allow_html=True)
        st.markdown("<p style='text-align: right; margin-top: 10px; font-size: 0.9rem;'>— Take a moment for yourself</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- Row 2: Quick Actions (Similar to the mobile app's icon buttons) ---
    st.markdown("<h2>Quick Actions</h2>")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # The buttons now inherit the smooth, rounded, purple styling from the custom CSS
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
    st.markdown("<h2>Your Toolkit</h2>")
    
    col_mood, col_journal = st.columns(2)

    with col_mood:
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #5D54A4; height: 180px;'>
            <h3 style='margin-top:0;'>Mood Tracker & Analysis 📈</h3>
            <p style='font-size: 0.95rem;'>Log your daily emotional state and see your personal timeline evolve. Earn badges for consistency.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\'📈 Mood Tracker\']').click(); return false;" style='color:#5D54A4; font-weight: 600; text-decoration: none;'>View Dashboard →</a></div>
        </div>
        """, unsafe_allow_html=True)

    with col_journal:
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #28A745; height: 180px;'>
            <h3 style='margin-top:0;'>Mindful Journaling 📝</h3>
            <p style='font-size: 0.95rem;'>A private space for reflection. The AI analyzes your entries to provide insights and sentiment scores.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\'📝 Mindful Journaling\']').click(); return false;" style='color:#28A745; font-weight: 600; text-decoration: none;'>Start Writing →</a></div>
        </div>
        """, unsafe_allow_html=True)
        
    # IoT/ECE Card
    with st.container():
        st.markdown(f"""
        <div class='card' style='border-left: 8px solid #FF5733;'>
            <h3 style='margin-top:0;'>IoT Monitoring (ECE Demo) ⚙️</h3>
            <p style='font-size: 0.95rem;'>Simulated real-time physiological data (Heart Rate/Stress) using a Kalman Filter demonstration.</p>
            <div style='text-align: right;'><a href='#' onclick="window.parent.document.querySelector('input[value=\'⚙️ IoT Dashboard (ECE)\']').click(); return false;" style='color:#FF5733; font-weight: 600; text-decoration: none;'>Go to Dashboard →</a></div>
        </div>
        """, unsafe_allow_html=True)


def mood_tracker_panel():
    st.header("Daily Mood Tracker 📈")
    
    with st.container():
        # Inner content of the mood tracker card
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
    with st.container(height=500, border=True):
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
    # TEMPORARILY DISABLED FUNCTION BODY TO AVOID SPAWN ERROR (Audio/System Dependency)
    st.header("Mindful Breathing 🧘‍♀️")
    st.markdown("---")
    st.error("⚠️ **This feature is temporarily disabled** due to a conflict with system dependencies during deployment. Please try again later.")
    st.markdown("Follow the prompts for 3 cycles: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Breathe deep and recenter.")
    
    # Placeholder for the actual breathing logic
    if st.button("Start Breathing (Disabled)", disabled=True, use_container_width=True):
         pass

    # Here is where the incomplete code was. I'm leaving it as a placeholder:
    # if "breathing_state" not in st.session_state:
    #     st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}

    # bs = st.session_state["breathing_state"]
    
    # # ... (rest of the original complex logic)
    
def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    st.markdown("Take a few minutes to write down your thoughts, feelings, or what happened today. This is for your eyes only.")
    
    with st.container():
        st.markdown("<div class='card' style='border-left: 5px solid #28A745;'>", unsafe_allow_html=True)
        new_entry = st.text_area("What's on your mind?", key="journal_text_area", height=200)
        
        if st.button("Save Entry & Analyze Sentiment", key="save_journal_btn"):
            if new_entry:
                sentiment = sentiment_compound(new_entry)
                entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    "text": new_entry, 
                    "sentiment": sentiment
                }
                st.session_state["daily_journal"].insert(0, entry)
                
                # Save to database if connected
                if st.session_state.get("_db_connected") and st.session_state.get("user_id"):
                    if save_journal_db(st.session_state["user_id"], new_entry, sentiment):
                        st.success("Entry saved and synced! Sentiment analyzed.")
                    else:
                        st.warning("Entry saved locally. DB sync failed.")
                else:
                    st.info("Entry saved locally (not synced). Sentiment analyzed.")
                
                # Clear the text area and rerun
                st.session_state["journal_text_area"] = ""
                st.rerun()
            else:
                st.warning("The journal entry is empty.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Recent Entries")
    
    for entry in st.session_state["daily_journal"][:3]:
        sentiment_label = "Positive" if entry['sentiment'] > 0.05 else ("Negative" if entry['sentiment'] < -0.05 else "Neutral")
        sentiment_color = "#28a745" if sentiment_label == "Positive" else ("#dc3545" if sentiment_label == "Negative" else "#ffc107")
        
        st.markdown(f"""
        <div class='card' style='padding: 15px; margin-bottom: 10px; border-left: 5px solid {sentiment_color};'>
            <p style='font-size: 0.8rem; color: #888;'>{entry['date']}</p>
            <p style='margin-top: 5px;'>{entry['text'][:200]}...</p>
            <p style='font-size: 0.9rem; font-weight: 600; color: {sentiment_color};'>Sentiment: {sentiment_label} ({entry['sentiment']:.2f})</p>
        </div>
        """, unsafe_allow_html=True)


def journal_analysis_panel():
    st.header("Journal Analysis 📊")
    st.markdown("Gain insights into your long-term emotional patterns from your entries and chat history.")
    
    all_text = get_all_user_text()
    if not st.session_state["daily_journal"] and not all_text.strip():
        st.info("Start journaling or chatting to generate your analysis!")
        return
    
    # Sentiment Over Time Plot
    st.subheader("Sentiment Trend")
    if st.session_state["daily_journal"]:
        df = pd.DataFrame(st.session_state["daily_journal"]).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date') 
        fig = px.line(df, x='date', y='sentiment', title="Journal Sentiment Over Time", markers=True,
                      color_discrete_sequence=['#5D54A4'])
        fig.update_layout(yaxis_range=[-1.0, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No saved journal entries for trend analysis.")

    # Word Cloud (DISABLED FOR DEPLOYMENT)
    st.subheader("Common Themes (Word Cloud)")
    
    wordcloud_fig = generate_wordcloud_figure_if_possible(all_text)
    if wordcloud_fig:
        # If the function were enabled and worked:
        # st.pyplot(wordcloud_fig)
        pass 
    else:
        st.info("Word Cloud feature is temporarily disabled due to deployment dependencies.")

    # AI Summary
    st.subheader("AI Insight Summary")
    if st.session_state.get("_ai_available"):
        summary_btn = st.button("Generate Summary of All Text", key="ai_summary_btn")
        if summary_btn:
            with st.spinner("AI is analyzing hundreds of thoughts..."):
                analysis_prompt = f"""
                Analyze the following collection of the user's journal entries and chat messages. 
                Identify the 3 most common themes, the average emotional tone (positive, negative, or mixed), 
                and provide one gentle, actionable suggestion for the user based on the content.
                ---
                Content: "{all_text[:5000]}..."
                """
                
                # Use a specific, short history for this analysis call to save tokens
                messages_for_analysis = [{"role": "system", "content": "You are a professional mood analyst. Provide a brief, bulleted analysis of the user's emotions and themes."}]
                messages_for_analysis.append({"role": "user", "content": analysis_prompt})
                
                try:
                    client = st.session_state["_ai_model"]
                    resp = client.chat.completions.create(
                        model=OPENROUTER_MODEL_NAME,
                        messages=messages_for_analysis,
                        max_tokens=500,
                        temperature=0.3
                    )
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(resp.choices[0].message.content)
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception:
                    st.error("Could not generate AI summary. Check API connection.")
    else:
        st.warning("AI is not available to generate a summary.")


def iot_dashboard_panel():
    st.header("IoT Dashboard (ECE Demo) ⚙️")
    st.markdown("Real-time simulated physiological data monitoring and Kalman filter demonstration.")
    
    placeholder = st.empty()

    if "sim_running" not in st.session_state:
        st.session_state["sim_running"] = False
    
    col_btn_sim, col_btn_reset = st.columns(2)
    with col_btn_sim:
        if st.session_state["sim_running"]:
            if st.button("Stop Simulation", key="stop_sim_btn", use_container_width=True):
                st.session_state["sim_running"] = False
                st.rerun()
        else:
            if st.button("Start Simulation", key="start_sim_btn", use_container_width=True):
                st.session_state["sim_running"] = True
                st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
                st.session_state["kalman_state"] = initialize_kalman()
                st.rerun()
    with col_btn_reset:
        if st.button("Reset Data", key="reset_data_btn", use_container_width=True):
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            st.rerun()


    if st.session_state["sim_running"]:
        with placeholder.container():
            st.subheader("Live Vitals")
            
            # 1. Generate new data
            current_time = int(time.time() * 1000)
            data_point = generate_simulated_physiological_data(current_time)
            
            # 2. Apply Kalman Filter
            hr_est, st.session_state["kalman_state"] = kalman_filter_simple(data_point["raw_ppg_signal"], st.session_state["kalman_state"])
            data_point["filtered_hr"] = hr_est

            # 3. Append to DataFrame (keeping only the last 100 points for performance)
            new_df = pd.DataFrame([data_point])
            st.session_state["physiological_data"] = pd.concat([st.session_state["physiological_data"], new_df], ignore_index=True).tail(100)
            
            df = st.session_state["physiological_data"]

            # 4. Display Metrics
            col_hr, col_stress = st.columns(2)
            col_hr.metric("Filtered Heart Rate (BPM)", f"{hr_est:.1f}")
            col_stress.metric("GSR Stress Level", f"{data_point['gsr_stress_level']:.2f}")

            # 5. Plotting (Heart Rate)
            fig_hr = px.line(df, x='time_ms', y=['raw_ppg_signal', 'filtered_hr'], 
                             title='Heart Rate: Raw vs. Kalman Filtered',
                             labels={'value': 'BPM', 'time_ms': 'Time (ms)'},
                             color_discrete_map={'raw_ppg_signal': '#E91E63', 'filtered_hr': '#5D54A4'})
            st.plotly_chart(fig_hr, use_container_width=True)

            # 6. Plotting (GSR)
            fig_gsr = px.line(df, x='time_ms', y='gsr_stress_level', 
                             title='GSR (Stress) Level',
                             labels={'value': 'Stress Index', 'time_ms': 'Time (ms)'},
                             color_discrete_sequence=['#FFC107'])
            st.plotly_chart(fig_gsr, use_container_width=True)

        time.sleep(0.5)
        st.rerun()
    elif not st.session_state["physiological_data"].empty:
        with placeholder.container():
             st.subheader("Last Session Summary")
             df = st.session_state["physiological_data"]
             max_hr = df['filtered_hr'].max()
             avg_hr = df['filtered_hr'].mean()
             max_gsr = df['gsr_stress_level'].max()
             
             col_hr, col_stress = st.columns(2)
             col_hr.metric("Avg HR (BPM)", f"{avg_hr:.1f}")
             col_stress.metric("Max Stress Index", f"{max_gsr:.2f}")
             
             st.info(f"Summary: Max Heart Rate reached {max_hr:.1f} BPM. Consider a mindful break!")
             
             fig_hr = px.line(df, x='time_ms', y=['raw_ppg_signal', 'filtered_hr'], title='Heart Rate: Raw vs. Kalman Filtered', color_discrete_map={'raw_ppg_signal': '#E91E63', 'filtered_hr': '#5D54A4'})
             st.plotly_chart(fig_hr, use_container_width=True)
    else:
        st.info("Press 'Start Simulation' to begin live physiological data monitoring.")


# Placeholder functions for PHQ-9 and Summary
def wellness_checkin_panel():
    st.header("Wellness Check-in (PHQ-9) 🩺")
    st.markdown("This check-in is based on the PHQ-9, a common screening tool for depression. Your answers are private.")
    
    PHQ9_QUESTIONS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself—or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed. Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead or of hurting yourself in some way"
    ]
    
    SCORE_MAPPING = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    current_answers = []
    
    for i, q in enumerate(PHQ9_QUESTIONS):
        answer = st.radio(f"{i+1}. Over the past two weeks, how often have you been bothered by the following problems: **{q}**", 
                          options=list(SCORE_MAPPING.keys()), 
                          key=f"phq9_q_{i}")
        current_answers.append(SCORE_MAPPING[answer])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Calculate Score & Interpretation", key="phq9_calculate_btn", type="primary"):
        total_score = sum(current_answers)
        
        if 0 <= total_score <= 4:
            interpretation = "Minimal Depression. You may not need treatment."
        elif 5 <= total_score <= 9:
            interpretation = "Mild Depression. Watchful waiting and repeat screening recommended."
        elif 10 <= total_score <= 14:
            interpretation = "Moderate Depression. Treatment plan recommended."
        elif 15 <= total_score <= 19:
            interpretation = "Moderately Severe Depression. Active treatment with medication and/or therapy recommended."
        else: # 20-27
            interpretation = "Severe Depression. Immediate treatment required."
            
        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation
        
        st.subheader("Your Results")
        st.metric("Total PHQ-9 Score", total_score)
        st.info(f"**Interpretation:** {interpretation}")
        
        st.session_state["phq9_last_completed"] = datetime.now().strftime("%Y-%m-%d")
        st.rerun()
    
    if st.session_state.get("phq9_score") is not None:
        st.subheader("Last Score Summary")
        st.metric("Score", st.session_state["phq9_score"])
        st.info(f"Interpretation: {st.session_state['phq9_interpretation']}")


def report_summary_panel():
    st.header("Report & Summary 📄")
    st.markdown("A consolidated view of your wellness metrics and progress.")
    
    col_score, col_last = st.columns(2)
    
    col_score.metric("Last PHQ-9 Score", st.session_state.get("phq9_score", "N/A"))
    col_last.metric("Last Mood Log", st.session_state["streaks"].get("last_mood_date", "N/A"))
    
    st.markdown("---")

    if st.session_state.get("phq9_interpretation"):
         st.subheader("Wellness Check-in Interpretation")
         st.info(f"**PHQ-9 Result:** {st.session_state['phq9_interpretation']}")
    
    # Simple Mood Summary
    st.subheader("Mood Trends")
    if st.session_state["mood_history"]:
        df = pd.DataFrame(st.session_state["mood_history"]).copy()
        avg_mood = df['mood'].mean()
        st.metric("Average Mood Score", f"{avg_mood:.2f}/11")
        
        st.subheader("All Time Journal Sentiment")
        all_sentiment = [e["sentiment"] for e in st.session_state["daily_journal"]]
        if all_sentiment:
            avg_sentiment = sum(all_sentiment) / len(all_sentiment)
            st.metric("Average Journal Sentiment", f"{avg_sentiment:.3f}")
        else:
            st.info("No journal sentiment recorded.")
            
        st.subheader("Your Badges")
        if st.session_state["streaks"]["badges"]:
            st.success(", ".join(st.session_state["streaks"]["badges"]))
        else:
            st.info("Keep logging to earn your first badge!")

    st.markdown("---")
    st.markdown("_This report is generated for your self-reflection and is not a substitute for professional medical advice._")


# ---------- Page Routing (Retained) ----------

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
    mindful_breathing_panel() # This is the disabled version now
elif st.session_state["page"] == "IoT Dashboard (ECE)":
    iot_dashboard_panel()
elif st.session_state["page"] == "Wellness Check-in":
    wellness_checkin_panel()
elif st.session_state["page"] == "Report & Summary":
    report_summary_panel()