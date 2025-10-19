import streamlit as st
import os
import time
import random
import re
import uuid
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components # CRITICAL: Imported for JS control

# Lightweight sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import the OpenAI library (used for OpenRouter compatibility)
from openai import OpenAI
from openai import APIError

# Placeholder for Supabase client
try:
    from supabase import create_client
except ImportError:
    # If the user hasn't installed supabase, we define a dummy client to prevent errors
    def create_client(*args, **kwargs):
        return None

# ---------- CONSTANTS ----------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" 
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" 
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
    10: "🥳 Joyful",
    11: "🌟 Fantastic" 
}

PHQ9_QUESTIONS = [
    "1. Little interest or pleasure in doing things?",
    "2. Feeling down, depressed, or hopeless?",
    "3. Trouble falling or staying asleep, or sleeping too much?",
    "4. Feeling tired or having little energy?",
    "5. Poor appetite or overeating?",
    "6. Feeling bad about yourself—or that you are a failure or have let yourself or your family down?",
    "7. Trouble concentrating on things, suchs as reading the newspaper or watching television?",
    "8. Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?",
    "9. Thoughts that you would be better off dead or of hurting yourself in some way?"
]

PHQ9_SCORES = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# CBT Prompts
CBT_PROMPTS = [
    "**1. The Situation:** What event or trigger led to the strong negative feeling?",
    "**2. The Emotion:** What emotion did you feel? (e.g., Sad, Angry, Anxious, Worthless, Lonely)",
    "**3. The Thought:** What specific automatic negative thought went through your mind? (This is the most crucial part!)",
    "**4. The Evidence FOR the thought:** What facts support your negative thought?",
    "**5. The Evidence AGAINST the thought:** What facts or alternative perspectives go against your negative thought? (Look for exceptions, logic, or other interpretations)",
    "**6. The Balanced Reframe:** What is a more helpful, realistic, and balanced thought you can have right now?"
]

# [Goals/Habits]
DEFAULT_GOALS = {
    "log_mood": {"name": "Log Mood", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None},
    "journal_entry": {"name": "Journal Entry", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None},
    "breathing_session": {"name": "Breathing Session", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None}
}

# ---------- Streamlit page config and LAYOUT SETUP (Modernized) ----------
st.set_page_config(
    page_title="HarmonySphere", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

def setup_page_and_layout():
    # --- NEW LILAC DREAM PALETTE CONSTANTS ---
    LILAC_PRIMARY = "#B0A3FF" # Main purple/lilac for accents, buttons
    SKY_BLUE_ACCENT = "#9DD6FF" # Main light blue/sky for softer accents
    DARK_TEXT = "#2C3E50" # Dark blue/gray for primary text
    BG_START = "#E6E6FA" # Lavender Blush (Lighter start)
    BG_END = "#ADD8E6"   # Light Blue (Darker end)
    GLASS_BORDER = "1px solid rgba(255, 255, 255, 0.4)"

    
    # Check if the user is logged in (Used for sidebar visibility logic)
    is_logged_in = st.session_state.get("logged_in", False)
    
    # --- CSS STYLING (Lilac Dream Theme & Glassmorphism) ---
    st.markdown(f"""
<style>

/* --- ANIMATIONS --- */
@keyframes sparkle {{
    0% {{ box-shadow: 0 0 5px {LILAC_PRIMARY}, 0 0 10px {SKY_BLUE_ACCENT}; }}
    50% {{ box-shadow: 0 0 15px {LILAC_PRIMARY}, 0 0 25px {SKY_BLUE_ACCENT}; }}
    100% {{ box-shadow: 0 0 5px {LILAC_PRIMARY}, 0 0 10px {SKY_BLUE_ACCENT}; }}
}}

@keyframes float-in {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes glow {{
    0% {{ box-shadow: 0 0 5px {LILAC_PRIMARY}, 0 0 0px {SKY_BLUE_ACCENT}; }}
    100% {{ box-shadow: 0 0 15px {LILAC_PRIMARY}, 0 0 20px {SKY_BLUE_ACCENT}; }}
}}

/* 1. Global Background and Typography */
.stApp {{ 
    background: linear-gradient(135deg, {BG_START}, {BG_END}); 
    color: {DARK_TEXT}; 
    font-family: 'Poppins', sans-serif; 
}}

/* Apply subtle float-in to main content blocks */
.main .block-container {{
    color: {DARK_TEXT} !important; 
    padding: 2rem 3rem;
    animation: float-in 0.8s ease-out;
}}


/* 2. Streamlit TextArea/Input fields (Glassmorphism) */
textarea, input[type="text"], input[type="email"] {{
    color: {DARK_TEXT} !important;
    -webkit-text-fill-color: {DARK_TEXT} !important;
    opacity: 1 !important;
    background: rgba(255, 255, 255, 0.15) !important; /* Semi-transparent white */
    backdrop-filter: blur(8px) !important; /* Glass effect */
    border: {GLASS_BORDER} !important; /* Light border */
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}}
textarea:focus, input[type="text"]:focus, input[type="email"]:focus {{
    border-color: {LILAC_PRIMARY} !important;
    box-shadow: 0 0 10px rgba(176, 163, 255, 0.5); /* Lilac shadow on focus */
}}

/* 3. Custom Card Style (Glassmorphism) */
.metric-card {{
    padding: 25px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.2); /* Semi-transparent */
    backdrop-filter: blur(10px); /* Glassmorphism effect */
    box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    border: {GLASS_BORDER}; /* Light, subtle border */
    transition: transform 0.4s, box-shadow 0.4s, background 0.4s;
    margin-bottom: 20px;
}}
.metric-card:hover {{
    transform: translateY(-8px); /* More noticeable float-lift */
    box-shadow: 0 15px 35px rgba(0,0,0,0.15), 0 0 20px {SKY_BLUE_ACCENT}; /* Floating + soft blue glow */
    cursor: pointer;
    background: rgba(255, 255, 255, 0.4);
}}

/* 4. Sidebar Styles */
[data-testid="stSidebar"] {{
    background: linear-gradient(to bottom, #FFFFFF, {BG_END}); 
    box-shadow: 2px 0 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease-in-out;
    {'visibility: hidden; transform: translateX(-100%); width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}
[data-testid="stSidebar"] > div:first-child {{
    {'width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}


/* 5. Primary Button Style (Lilac Glow) */
.stButton>button {{
    color: {DARK_TEXT}; /* Dark text for contrast */
    background: {LILAC_PRIMARY}; 
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    transition: all 0.3s;
}}
.stButton>button:hover {{
    background: #C3B9FF; /* Slightly lighter Lilac on hover */
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15), 0 0 25px {LILAC_PRIMARY}; /* Glow effect */
}}

/* Ensure button text is visible (Lilac background + dark text) */
.stButton>button>div, .stButton>button span p {{
    color: {DARK_TEXT} !important;
}}

/* 6. Sidebar Status Tags */
.sidebar-status {{
    padding: 6px 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
}}
.status-connected {{ background-color: #D4EDDA; color: {DARK_TEXT}; border-left: 4px solid #28A745; }}
.status-local {{ background-color: #FFF3CD; color: {DARK_TEXT}; border-left: 4px solid #FFC107; }}

/* 7. Hide Streamlit Footer */
footer {{
    visibility: hidden;
}}


/* Breathing Effect CSS (Lilac/Sky Blue) */
@keyframes pulse-in {{
    0% {{ transform: scale(0.6); opacity: 0.8; }} 
    100% {{ transform: scale(1.0); opacity: 1.0; }}
}}
@keyframes pulse-out {{
    0% {{ transform: scale(1.0); opacity: 1.0; }}
    100% {{ transform: scale(0.6); opacity: 0.8; }}
}}

.heart-shape {{
    position: relative;
    width: 250px; 
    height: 250px; 
    background-color: {LILAC_PRIMARY}; 
    transform: rotate(-45deg);
    border-radius: 0 50% 0 0; 
    box-shadow: 0 0 50px rgba(176, 163, 255, 0.7); /* Lilac shadow */
    transition: background-color 0.5s;
}}

.heart-shape::before,
.heart-shape::after {{
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    background-color: {LILAC_PRIMARY}; 
    border-radius: 50%;
    transition: background-color 0.5s; 
}}

.heart-shape::before {{
    top: -125px; 
    left: 0;
}}

.heart-shape::after {{
    top: 0;
    left: 125px; 
}}

/* FIX: Z-INDEX FOR TEXT VISIBILITY */
.breathing-heart-content {{
    position: relative; 
    z-index: 10; /* CRITICAL FIX: Forces text above heart layers */
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(45deg); 
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: white; /* Text color remains white for contrast */
    font-size: 1.5rem;
    font-weight: bold;
    width: 100%;
    height: 100%;
}}

/* Parent container for the animation */
.heart-animation-wrapper {{
    position: relative;
    width: 350px; 
    height: 350px;
    margin: 50px auto;
    /* SPARKLE/SHIMMER EFFECT */
    animation: sparkle 3s infinite alternate ease-in-out;
}}

/* Specific state animations */
.heart-animation-wrapper.inhale {{
    animation: pulse-in 4s ease-in-out forwards; 
}}
.heart-animation-wrapper.hold {{
    transform: scale(1.0); 
    animation-duration: 7s; 
}}
.heart-animation-wrapper.exhale {{
    animation: pulse-out 8s ease-in-out forwards; 
}}

/* Adjust colors of the heart shape based on state */
.heart-animation-wrapper.inhale .heart-shape,
.heart-animation-wrapper.inhale .heart-shape::before,
.heart-animation-wrapper.inhale .heart-shape::after {{
    background-color: {LILAC_PRIMARY}; 
}}
.heart-animation-wrapper.hold .heart-shape,
.heart-animation-wrapper.hold .heart-shape::before,
.heart-animation-wrapper.hold .heart-shape::after {{
    background-color: #8A7DCB; /* Darker Lilac for Hold */
}}
.heart-animation-wrapper.exhale .heart-shape,
.heart-animation-wrapper.exhale .heart-shape::before,
.heart-animation-wrapper.exhale .heart-shape::after {{
    background-color: {SKY_BLUE_ACCENT}; /* Sky Blue for Exhale */
}}

</style>
""", unsafe_allow_html=True)

# Call the setup function early in the main script flow
setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------
@st.cache_data
def initialize_kalman(Q_val=0.01, R_val=0.1):
    P = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    x = np.array([0.0, 0.0]) 
    F = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    Q = np.array([[Q_val, 0.0], [0.0, Q_val]]) 
    H = np.array([[1.0, 0.0]]) 
    R = np.array([[R_val]]) 
    return {'x': x, 'P': P, 'F': F, 'Q': Q, 'H': H, 'R': R}

def kalman_filter_simple(z_meas, state):
    x, P, F, Q, H, R = state['x'], state['P'], state['F'], state['Q'], state['H'], state['R']

    # Prediction
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Update
    y = z_meas - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = P_pred - K @ H @ P_pred

    state.update({'x': x, 'P': P})
    return x[0], state

def generate_simulated_physiological_data(current_time_ms):
    base_hr = 70
    t = current_time_ms / 1000.0 
    noise = np.random.randn() * 0.5
    sin_wave = 5 * np.sin(t / 10.0) 
    hr_raw = base_hr + sin_wave + noise
    
    if 30 <= t % 60 < 60:
        hr_raw += 15 * np.exp(-(t % 60 - 30) / 5) 

    base_rr = 15
    rr_raw = base_rr + np.sin(t / 5.0) * 2 + np.random.randn() * 0.2
    
    hr_raw = max(60, hr_raw)
    rr_raw = max(10, rr_raw)
    
    return hr_raw, rr_raw


# ---------- CACHING & LAZY SETUP ----------
analyzer = None
try:
    analyzer = SentimentIntensityAnalyzer()
except NameError:
    pass

supabase_client = None 

@st.cache_resource
def setup_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def setup_ai_model(api_key: str, history: list):
    try:
        if not api_key:
            return None
        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"AI setup failed: {e}")
        return None
    
@st.cache_resource(show_spinner=False)
def setup_supabase_client(url: str, key: str):
    if url and key:
        try:
            return create_client(url, key)
        except Exception as e:
            st.error(f"Supabase connection failed: {e}")
            return None
    return None

@st.cache_resource(show_spinner=False)
def get_supabase_admin_client():
    return st.session_state.supabase_client

# ---------- Session state defaults (CLEANED UP) ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "show_splash" not in st.session_state:
    st.session_state["show_splash"] = True
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Guest"
if "daily_progress" not in st.session_state:
    st.session_state["daily_progress"] = 70
if "journal_entries" not in st.session_state:
    st.session_state["journal_entries"] = []
if "cbt_records" not in st.session_state:
    st.session_state["cbt_records"] = []
if "mood_logs" not in st.session_state:
    st.session_state["mood_logs"] = pd.DataFrame(columns=['date', 'mood_level', 'note'])
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "system", "content": "You are a supportive, non-judgemental wellness and mental health coach named Harmony. Keep responses encouraging and brief."}]
if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop" 
if "is_breathing_active" not in st.session_state: 
    st.session_state["is_breathing_active"] = False
if "phq9_history" not in st.session_state:
    st.session_state["phq9_history"] = []
if "goals" not in st.session_state:
    st.session_state["goals"] = DEFAULT_GOALS
if "hr_kalman_state" not in st.session_state:
    st.session_state["hr_kalman_state"] = initialize_kalman(Q_val=0.005, R_val=0.2)
if "rr_kalman_state" not in st.session_state:
    st.session_state["rr_kalman_state"] = initialize_kalman(Q_val=0.005, R_val=0.2)
if "supabase_client" not in st.session_state:
    st.session_state["supabase_client"] = setup_supabase_client(
        os.getenv("SUPABASE_URL"), 
        os.getenv("SUPABASE_KEY")
    )
if "ai_client" not in st.session_state:
    st.session_state["ai_client"] = setup_ai_model(os.getenv("OPENROUTER_API_KEY"), st.session_state.chat_history)
if "current_hr" not in st.session_state:
    st.session_state["current_hr"] = 70.0
if "current_rr" not in st.session_state:
    st.session_state["current_rr"] = 15.0

# ---------- AI/Sentiment Helper functions (All preserved) ----------
def clean_text_for_ai(text: str) -> str:
    return text.replace("\n", " ").strip()

def safe_generate(prompt: str, max_tokens: int = 300):
    client = st.session_state.ai_client
    if not client:
        return "AI is not configured. Please check API key."
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL_NAME,
            messages=[
                {"role": "system", "content": st.session_state.chat_history[0]['content']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except APIError as e:
        return f"AI API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def sentiment_compound(text: str) -> float:
    analyzer = setup_analyzer()
    if analyzer:
        vs = analyzer.polarity_scores(text)
        return vs['compound']
    return 0.0

# -------------------- FRONT-END COMPONENTS --------------------

def app_splash_screen():
    # Use the new theme colors in the splash screen
    st.markdown(f"""
    <style>
    .splash-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        width: 100vw;
        background: linear-gradient(135deg, #E6E6FA, #ADD8E6);
        color: #2C3E50;
        animation: fade-out 2s 1s forwards;
    }}
    .splash-title {{
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 20px;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px rgba(176, 163, 255, 0.5);
    }}
    .splash-info {{
        font-size: 1.2rem;
    }}
    @keyframes fade-out {{
        0% {{ opacity: 1; }}
        100% {{ opacity: 0; display: none; }}
    }}
    </style>
    <div class="splash-container">
        <div class="splash-title">HarmonySphere ✨</div>
        <div class="splash-info">Loading your personal wellness ecosystem...</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3) # Increased sleep for smoother transition based on user request
    st.session_state["show_splash"] = False
    st.rerun()

def unauthenticated_home():
    # Center the login panel
    st.markdown("<div style='display:flex; justify-content:center; align-items:center; height: 80vh;'>", unsafe_allow_html=True)
    with st.container(border=False):
        st.title("Welcome to HarmonySphere")
        st.subheader("Login to start your dreamy journey.")

        # Apply glassmorphism to the login form container
        with st.container(border=False):
            st.markdown(f"""
            <div class='metric-card' style='padding: 30px; max-width: 400px; margin: auto;'>
                <p style='font-weight: 600; font-size: 1.1rem; color: #2C3E50;'>Enter your details</p>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                email = st.text_input("Email (e.g., user@example.com)")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login / Register", type="primary")

                if submitted:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                        return
                    # Simplified dummy auth logic
                    st.session_state["logged_in"] = True
                    st.session_state["user_name"] = email.split('@')[0]
                    st.session_state["user_id"] = str(uuid.uuid4()) 
                    st.success(f"Welcome, {st.session_state['user_name']}!")
                    time.sleep(1)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_navigation():
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.user_name}!")
        
        total_goals = len(st.session_state.goals)
        completed_goals = sum(1 for goal in st.session_state.goals.values() if goal['count'] >= goal['target'])
        
        if total_goals > 0:
            progress_val = int((completed_goals / total_goals) * 100)
        else:
            progress_val = 0
            
        st.subheader("Daily Goal Progress")
        st.progress(progress_val, text=f"{completed_goals} / {total_goals} Goals Completed")
        st.markdown(f"**Focus on your goals today**")
        
        # Wellness Check-in Box (using metric-card style)
        st.markdown(f"""
        <div class='metric-card' style='background: rgba(255, 255, 255, 0.8); padding: 10px; border-left: 5px solid #9DD6FF;'>
            <p style='color: #2C3E50;'>Complete a Wellness Check-in! ✨</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Navigation Buttons
        nav_pages = {
            "Home": "🏠 Dashboard",
            "Mindful Breathing": "🧘 Breathing Exercise",
            "Mood Tracker": "😊 Mood Tracker",
            "Mindful Journaling": "📝 Journaling",
            "CBT Thought Record": "🧠 CBT Thought Record",
            "Journal Analysis": "📊 Journal Analysis",
            "Report & Summary": "📈 Report & Summary",
            "IoT Dashboard (ECE)": "🔌 IoT Dashboard (ECE)",
            "AI Chat": "🤖 AI Coach",
        }

        for page_key, page_title in nav_pages.items():
            if st.button(page_title, key=f"nav_{page_key}", use_container_width=True): 
                st.session_state["page"] = page_key
                st.rerun()


def homepage_panel():
    st.header("🏠 Dashboard Overview")
    st.success(f"Welcome back, {st.session_state.user_name}! Let's find your balance.")
    st.info(random.choice(QUOTES))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Note: The metric-card CSS now handles the glassmorphism
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Current Mood</p>
            <h2 style='margin: 0.2em 0;'>{MOOD_EMOJI_MAP.get(st.session_state.mood_logs['mood_level'].iloc[-1] if not st.session_state.mood_logs.empty else 7)}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Latest Log</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Journal Entries</p>
            <h2 style='margin: 0.2em 0;'>{len(st.session_state.journal_entries)}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Total Recorded</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Daily Progress</p>
            <h2 style='margin: 0.2em 0;'>{st.session_state.daily_progress}%</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Completed Goals</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Placeholder for Mood Trend Chart
    st.subheader("Your Mood Trend (Last 7 Days)")
    if not st.session_state.mood_logs.empty:
        mood_df = st.session_state.mood_logs.copy()
        mood_df['date'] = pd.to_datetime(mood_df['date'])
        mood_df.set_index('date', inplace=True)
        mood_df = mood_df.tail(7)
        
        # Use Lilac for chart line
        fig = px.line(mood_df, y='mood_level', title="Mood Over Time", 
                      labels={'mood_level': 'Mood Level (1-10)'},
                      markers=True, line_shape='spline',
                      color_discrete_sequence=["#B0A3FF"]) # Use LILAC_PRIMARY hex
        fig.update_layout(yaxis_range=[1, 10])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your mood to see your personal trend here!")


def mindful_breathing_page():
    st.header("🧘 Mindful Breathing Exercise")
    st.write("Follow the expansion and contraction of the shimmering heart. The smooth transition is now active!")

    if st.session_state.get("is_breathing_active"):
        
        # --- CRITICAL: THE JAVASCRIPT ANIMATION LOGIC IS ADDED HERE ---
        
        animation_html = f"""
        <div id="animation-container">
            <div class="heart-animation-wrapper stop">
                <div class="heart-shape">
                    <div class="breathing-heart-content">
                        <span id="breathing-text">Ready...</span>
                    </div>
                </div>
            </div>
            <div id="countdown-timer-display" style="text-align:center; font-size: 1.2rem; font-weight: bold; margin-top: 20px; color: #2C3E50;"></div>
        </div>
        <script>
            // Ensure the script only runs once by checking if elements exist
            if (!document.getElementById('breathing-text').hasAttribute('data-initialized')) {{

                const wrapper = document.querySelector('.heart-animation-wrapper');
                const textSpan = document.getElementById('breathing-text');
                const timerDisplay = document.getElementById('countdown-timer-display');
                const DURATION_SECONDS = 300; // 5 minutes
                
                // Breathing Cycle Times (milliseconds)
                const INHALE_TIME = 4000; 
                const HOLD_TIME = 2000; 
                const EXHALE_TIME = 7000; 
                const CYCLE_TIME = INHALE_TIME + HOLD_TIME + EXHALE_TIME;

                let totalRemaining = DURATION_SECONDS;
                let cycleInterval;
                let timerInterval;

                // Mark as initialized
                textSpan.setAttribute('data-initialized', 'true');

                function updateCountdown() {{
                    if (totalRemaining > 0) {{
                        const minutes = Math.floor(totalRemaining / 60);
                        const seconds = totalRemaining % 60;
                        timerDisplay.textContent = `Time Remaining: ${{minutes}}:${{seconds < 10 ? '0' : ''}}${{seconds}}`;
                        totalRemaining--;
                    }} else {{
                        // Stop all timers and set completion state
                        clearInterval(timerInterval);
                        clearInterval(cycleInterval);
                        textSpan.textContent = "DONE!";
                        wrapper.className = "heart-animation-wrapper stop";
                        timerDisplay.textContent = "Session Complete! Great Job!";
                        
                        // Use Streamlit custom event to notify Python state to change
                        // This uses an iframe-safe method to communicate completion
                        const completeMessage = {{"streamlit_message": "breathing_complete"}};
                        window.parent.postMessage(completeMessage, "*");

                    }}
                }}
                
                function runCycle() {{
                    // 1. Inhale
                    wrapper.className = "heart-animation-wrapper inhale";
                    textSpan.textContent = "INHALE";

                    setTimeout(() => {{
                        // 2. Hold
                        wrapper.className = "heart-animation-wrapper hold";
                        textSpan.textContent = "HOLD";

                        setTimeout(() => {{
                            // 3. Exhale
                            wrapper.className = "heart-animation-wrapper exhale";
                            textSpan.textContent = "EXHALE";
                        }}, HOLD_TIME);
                        
                    }}, INHALE_TIME);
                }}

                // Start the breathing and the timer after a short delay
                setTimeout(() => {{
                    updateCountdown();
                    timerInterval = setInterval(updateCountdown, 1000);
                    runCycle(); 
                    cycleInterval = setInterval(runCycle, CYCLE_TIME);
                }}, 1000);
            }}
        </script>
        """
        
        # Use Streamlit components to inject HTML/JS.
        components.html(animation_html, height=500)
        
        # Listen for the JS message signaling completion
        if st.session_state.get("js_message") == "breathing_complete":
            st.session_state["is_breathing_active"] = False
            st.session_state.goals["breathing_session"]["count"] += 1
            st.balloons()
            st.toast("Mindful Breathing session completed and logged!")
            # Reset the message flag
            del st.session_state["js_message"]
            st.rerun() 


        # Stop button for the running animation
        if st.button("Stop Session", key="stop_breathing_button", type="secondary"):
            st.session_state["is_breathing_active"] = False
            st.warning("Session stopped.")
            st.rerun()
            
    else:
        # Static heart display and start button
        st.markdown(f"""
        <div class="heart-animation-wrapper stop">
            <div class="heart-shape">
                <div class="breathing-heart-content">
                    <span id="breathing-text">Start</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Action Log")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💖 Saved!", key="breathing_like_button", use_container_width=True, type="secondary"):
                st.session_state.goals["breathing_session"]["count"] += 1
                st.toast("Exercise logged!")
                
        with col2:
            # Button to trigger the animation logic
            if st.button("Start 5 Min Session", type="primary", use_container_width=True): 
                st.session_state["is_breathing_active"] = True
                st.info("Starting Mindful Breathing...")
                # Add a script tag to receive the JS completion message
                st.markdown("""
                <script>
                    window.addEventListener('message', event => {
                        if (event.data.streamlit_message === 'breathing_complete') {
                            const event = new CustomEvent('streamlit:setSessionState', {
                                detail: {
                                    key: 'js_message',
                                    value: 'breathing_complete'
                                }
                            });
                            window.parent.document.dispatchEvent(event);
                        }
                    });
                </script>
                """, unsafe_allow_html=True)
                st.rerun() # Reruns to switch to the active animation view


def mindful_journaling_page():
    st.header("📝 Mindful Journaling")
    
    with st.form("journal_form"):
        entry = st.text_area("What is on your mind today? (Max 1000 words)", height=300, max_chars=1000, key="journal_entry_text")
        save_button = st.form_submit_button("Save Entry", type="primary")

        if save_button and entry:
            sentiment = sentiment_compound(entry)
            st.session_state.journal_entries.append({
                "id": str(uuid.uuid4()),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": entry,
                "sentiment": sentiment
            })
            st.session_state.goals["journal_entry"]["count"] += 1
            st.success("Journal entry saved successfully!")
            
def mood_tracker_page():
    st.header("😊 Mood Tracker")
    
    mood_level = st.slider(
        "How are you feeling right now? (1 = Low, 10 = High)", 
        1, 10, 5, 
        format=f"%d - {MOOD_EMOJI_MAP.get(st.session_state.get('mood_temp_val', 5))}", 
        key='mood_temp_val'
    )
    
    mood_note = st.text_area("Optional: Add a note about why you feel this way (e.g., I had a great chat with a friend).", key="mood_note_text")
    
    if st.button("Log Mood", type="primary"):
        st.session_state.mood_logs = pd.concat([
            st.session_state.mood_logs,
            pd.DataFrame([{
                'date': datetime.now(), 
                'mood_level': mood_level, 
                'note': mood_note
            }])
        ], ignore_index=True)
        
        st.session_state.goals["log_mood"]["count"] += 1
        st.success(f"Mood logged successfully: {mood_level} {MOOD_EMOJI_MAP.get(mood_level)}")
        

def wellness_checkin_page():
    st.header("✅ PHQ-9 Depression Screening")
    st.info("This is a simple tool based on the Patient Health Questionnaire (PHQ-9). It is NOT a diagnostic tool. Please consult a professional for diagnosis.")
    
    responses = {}
    with st.form("phq9_form"):
        for i, question in enumerate(PHQ9_QUESTIONS):
            responses[question] = st.radio(
                question, 
                options=list(PHQ9_SCORES.keys()), 
                key=f"phq9_q_{i}"
            )
            
        submitted = st.form_submit_button("Calculate Score", type="primary")

        if submitted:
            total_score = sum(PHQ9_SCORES[response] for response in responses.values())
            
            if total_score <= 4:
                interpretation = "Minimal depression (Score 0-4)"
            elif total_score <= 9:
                interpretation = "Mild depression (Score 5-9)"
            elif total_score <= 14:
                interpretation = "Moderate depression (Score 10-14)"
            elif total_score <= 19:
                interpretation = "Moderately severe depression (Score 15-19)"
            else:
                interpretation = "Severe depression (Score 20-27)"
                
            st.markdown("---")
            st.subheader(f"Your PHQ-9 Score: {total_score}")
            st.warning(f"Interpretation: {interpretation}")
            
            st.session_state.phq9_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "score": total_score,
                "interpretation": interpretation
            })
            

def cbt_thought_record_page():
    st.header("🧠 CBT Thought Record")
    st.info("Use this tool to analyze and restructure automatic negative thoughts (ANTs).")
    
    record = {}
    with st.form("cbt_form"):
        for i, prompt in enumerate(CBT_PROMPTS):
            st.markdown(prompt) 
            record[prompt] = st.text_area(f"Response to Step {i+1}", key=f"cbt_step_{i}", height=70)
            
        submitted = st.form_submit_button("Review Thought Record", type="primary")
        
        if submitted:
            if not all(record.values()):
                st.error("Please fill out all sections of the Thought Record.")
                return
            
            cbt_entry = {
                "id": str(uuid.uuid4()),
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data": record
            }
            st.session_state.cbt_records.append(cbt_entry)
            st.success("Thought Record saved for review!")
            
            cbt_prompt = f"Analyze this CBT thought record and provide a brief, supportive summary and a single, powerful balanced thought:\nSituation: {record[CBT_PROMPTS[0]]}\nThought: {record[CBT_PROMPTS[2]]}\nEvidence For: {record[CBT_PROMPTS[3]]}\nEvidence Against: {record[CBT_PROMPTS[4]]}"
            
            with st.spinner("Asking Harmony to analyze your thought..."):
                analysis = safe_generate(cbt_prompt, max_tokens=250)
                st.markdown("---")
                st.subheader("🤖 Harmony's Analysis")
                st.write(analysis)


def journal_analysis_page():
    st.header("📊 Journal Analysis")
    st.info("See insights on the emotional tone of your past journal entries.")
    
    if not st.session_state.journal_entries:
        st.warning("No journal entries found. Start journaling to see analysis.")
        return

    journal_df = pd.DataFrame(st.session_state.journal_entries)
    journal_df['date'] = pd.to_datetime(journal_df['date']).dt.date
    
    st.subheader("Sentiment Trend")
    sentiment_fig = px.line(
        journal_df, x='date', y='sentiment', 
        title="Journal Sentiment Over Time",
        labels={'sentiment': 'Sentiment Score (-1 to 1)', 'date': 'Date'},
        markers=True, line_shape='linear',
        color_discrete_sequence=["#B0A3FF"] # Lilac color
    )
    st.plotly_chart(sentiment_fig, use_container_width=True)
    
    st.subheader("Common Themes")
    st.code("Word Cloud and key theme extraction would appear here.", language="markdown")
    st.write("Words like 'stressed', 'tired', and 'busy' are common in your recent entries.")


def report_summary_page():
    st.header("📈 Report & Summary")
    st.info("Your personalized mental wellness report.")
    
    st.subheader("Key Takeaways")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("Average Mood", f"{st.session_state.mood_logs['mood_level'].mean():.1f}" if not st.session_state.mood_logs.empty else "N/A")
    with col_b:
        st.metric("Average Sentiment", f"{pd.DataFrame(st.session_state.journal_entries)['sentiment'].mean():.2f}" if st.session_state.journal_entries else "N/A")
    
    if st.button("Download Full Report (PDF)", type="primary"):
        st.success("Report generation simulated! (PDF would be created here)")

def ai_chat_page():
    st.header("🤖 AI Wellness Coach - Harmony")
    
    for message in st.session_state.chat_history[1:]: 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Harmony for advice, support, or a reflection exercise..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Harmony is thinking..."):
                ai_response = safe_generate(prompt, max_tokens=350)
                st.markdown(ai_response)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})


def iot_dashboard_page():
    st.header("🔌 IoT Dashboard (ECE Demo)")
    st.info("Real-time physiological data simulated and filtered using a Kalman Filter. Notice the subtle shimmer on the metrics cards!")
    
    if st.button("Simulate Data Refresh", key="refresh_iot", type="secondary"):
        st.session_state["last_refresh"] = datetime.now()
        hr_raw, rr_raw = generate_simulated_physiological_data(int(time.time() * 1000))
        
        hr_filtered, st.session_state["hr_kalman_state"] = kalman_filter_simple(
            hr_raw, st.session_state["hr_kalman_state"]
        )
        rr_filtered, st.session_state["rr_kalman_state"] = kalman_filter_simple(
            rr_raw, st.session_state["rr_kalman_state"]
        )
        
        st.session_state["current_hr"] = hr_filtered
        st.session_state["current_rr"] = rr_filtered
        
        st.balloons() 
        
    st.markdown("---")

    col_hr, col_rr = st.columns(2)

    with col_hr:
        st.markdown(f"""
        <div class='metric-card' style='animation: sparkle 3s infinite alternate ease-in-out;'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Heart Rate (BPM)</p>
            <h2 style='margin: 0.2em 0; color: #B0A3FF;'>{st.session_state.current_hr:.1f}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Filtered by Kalman</p>
        </div>
        """, unsafe_allow_html=True)

    with col_rr:
        st.markdown(f"""
        <div class='metric-card' style='animation: sparkle 3s infinite alternate ease-in-out;'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Respiration Rate (BPM)</p>
            <h2 style='margin: 0.2em 0; color: #9DD6FF;'>{st.session_state.current_rr:.1f}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Filtered by Kalman</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Live Data Stream (Simulated)")
    st.warning("A real-time chart would update here. Click 'Simulate Data Refresh' to update the metrics.")
        
        
# -------------------- MAIN APP EXECUTION --------------------
# Main placeholder for content
app_placeholder = st.empty()

# ---------- MAIN APPLICATION LOGIC (Triple Flow) ----------
with app_placeholder.container():
    
    if st.session_state.get("show_splash"):
        app_splash_screen()
        
    elif not st.session_state.get("logged_in"):
        unauthenticated_home()

    else:
        # Load Sidebar Navigation
        sidebar_navigation()
        
        current_page = st.session_state["page"]
        
        if current_page == "Home":
            homepage_panel()
        elif current_page == "Mindful Journaling":
            mindful_journaling_page()
        elif current_page == "Mood Tracker":
            mood_tracker_page()
        elif current_page == "Wellness Check-in":
            wellness_checkin_page()
        elif current_page == "AI Chat":
            ai_chat_page() 
        elif current_page == "Mindful Breathing":
            mindful_breathing_page()
        elif current_page == "CBT Thought Record":
            cbt_thought_record_page()
        elif current_page == "Journal Analysis":
            journal_analysis_page()
        elif current_page == "Report & Summary": 
            report_summary_page()
        elif current_page == "IoT Dashboard (ECE)": 
            iot_dashboard_page()
        else:
            st.warning("Page not found or not yet implemented.")
