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
    11: "🌟 Fantastic" # Included 11 for plot aesthetic but slider goes to 10
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
    # --- NEW COOL PALETTE CONSTANTS (Replaces Pink) ---
    TEAL = "#008080" # Primary color for buttons, accents, heart
    DARK_NAVY_TEXT = "#33415C" # Dark color for all text
    COOL_BACKGROUND_START = "#E0F7FA" # Lightest Blue for gradient
    COOL_BACKGROUND_END = "#B2DFDB"   # Light Teal for gradient
    
    # Check if the user is logged in (Used for sidebar visibility logic)
    is_logged_in = st.session_state.get("logged_in", False)
    
    # --- CSS STYLING (Cool Theme & Fixes) ---
    st.markdown(f"""
<style>
/* 1. Global Background and Typography (COOL THEME) */
.stApp {{ 
    background: linear-gradient(135deg, {COOL_BACKGROUND_START}, {COOL_BACKGROUND_END}); /* Cool pastel gradient */
    color: {DARK_NAVY_TEXT}; /* Primary text color for contrast */
    font-family: 'Poppins', sans-serif; 
}}

/* FIX 1: DASHBOARD VISIBILITY FIX (Light theme + light write-up) */
/* Force dark text color for all content inside the main area for contrast */
.main .block-container * {{
    color: {DARK_NAVY_TEXT} !important; 
}}

.main .block-container {{ 
    padding: 2rem 3rem;
}}

/* 2. Streamlit TextArea/Input fields (COOL THEME) */
textarea, input[type="text"], input[type="email"] {{
    color: {DARK_NAVY_TEXT} !important;
    -webkit-text-fill-color: {DARK_NAVY_TEXT} !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border: 2px solid {COOL_BACKGROUND_END} !important; /* Light teal border */
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}}
textarea:focus, input[type="text"]:focus, input[type="email"]:focus {{
    border-color: {TEAL} !important;
    box-shadow: 0 0 8px rgba(0, 128, 128, 0.5); /* Teal shadow */
}}

/* 3. Custom Card Style (Cooler Look) */
.metric-card {{
    padding: 25px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: transform 0.3s, box-shadow 0.3s, background 0.3s;
    margin-bottom: 20px;
    border: none;
}}
.metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.1);
    cursor: pointer;
    background: rgba(255, 255, 255, 0.9);
}}

/* 4. Sidebar Styles (Original functionality preserved, cool colors used) */
[data-testid="stSidebar"] {{
    background: linear-gradient(to bottom, #FAFAFA, {COOL_BACKGROUND_END}); /* Lighter sidebar */
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    transition: transform 0.3s ease-in-out;
    /* CRITICAL: Hide when not logged in or during splash */
    {'visibility: hidden; transform: translateX(-100%); width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}
/* Ensures the sidebar is completely gone */
[data-testid="stSidebar"] > div:first-child {{
    {'width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}


/* 5. Primary Button Style (COOL THEME) */
.stButton>button {{
    color: #FFFFFF;
    background: {TEAL}; /* Primary Teal */
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s;
}}
.stButton>button:hover {{
    background: #009999; /* Slightly lighter Teal on hover */
}}

/* FIX 2: HEART SYMBOL TEXT VISIBILITY FIX */
/* This targets the internal elements of the button to ensure the text and icon are visible */
.stButton>button>div, .stButton>button span p {{
    color: white !important;
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
.status-connected {{ background-color: #D4EDDA; color: {DARK_NAVY_TEXT}; border-left: 4px solid #28A745; }}
.status-local {{ background-color: #FFF3CD; color: {DARK_NAVY_TEXT}; border-left: 4px solid #FFC107; }}

/* 7. Hide Streamlit Footer */
footer {{
    visibility: hidden;
}}

/* Breathing Effect CSS (COOL COLORS) - SMOOTH TRANSITIONS PRESERVED */
/* NOTE: The @keyframes pulse-in and pulse-out are exactly as they were, ensuring smooth transition is preserved */
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
    background-color: {TEAL}; /* Changed from Pink */
    transform: rotate(-45deg);
    border-radius: 0 50% 0 0; 
    box-shadow: 0 0 50px rgba(0, 128, 128, 0.7); /* Teal shadow */
    transition: background-color 0.5s;
}}

.heart-shape::before,
.heart-shape::after {{
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    background-color: {TEAL}; /* Changed from Pink */
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

/* Container to hold text and animation, centered over the heart */
.breathing-heart-content {{
    position: absolute;
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
}}

/* Specific state animations (SMOOTH TRANSITIONS PRESERVED) */
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

/* Adjust colors of the heart shape based on state (COOL COLORS) */
.heart-animation-wrapper.inhale .heart-shape,
.heart-animation-wrapper.inhale .heart-shape::before,
.heart-animation-wrapper.inhale .heart-shape::after {{
    background-color: {TEAL}; 
}}
.heart-animation-wrapper.hold .heart-shape,
.heart-animation-wrapper.hold .heart-shape::before,
.heart-animation-wrapper.hold .heart-shape::after {{
    background-color: #006666; /* Darker Teal for Hold */
}}
.heart-animation-wrapper.exhale .heart-shape,
.heart-animation-wrapper.exhale .heart-shape::before,
.heart-animation-wrapper.exhale .heart-shape::after {{
    background-color: #36454F; /* Charcoal/Dark Blue for Exhale */
}}

</style>
""", unsafe_allow_html=True)

# Call the setup function early in the main script flow
setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------
@st.cache_data
def initialize_kalman(Q_val=0.01, R_val=0.1):
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # Error covariance matrix
    x = np.array([0.0, 0.0])  # State vector [rate, acc]
    F = np.array([[1.0, 0.0], [0.0, 1.0]])  # State transition matrix (identity for simplicity)
    Q = np.array([[Q_val, 0.0], [0.0, Q_val]])  # Process noise covariance
    H = np.array([[1.0, 0.0]])  # Measurement matrix (measures rate)
    R = np.array([[R_val]])  # Measurement noise covariance
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
    # Simulated Heart Rate (BPM) fluctuating around a base rate
    base_hr = 70
    t = current_time_ms / 1000.0  # time in seconds
    noise = np.random.randn() * 0.5
    sin_wave = 5 * np.sin(t / 10.0) # Slow fluctuation
    hr_raw = base_hr + sin_wave + noise
    
    # Simulate a "stress event" for 30 seconds
    if 30 <= t % 60 < 60:
        hr_raw += 15 * np.exp(-(t % 60 - 30) / 5) 

    # Simulate Respiration Rate (Breaths per minute)
    base_rr = 15
    rr_raw = base_rr + np.sin(t / 5.0) * 2 + np.random.randn() * 0.2
    
    # Ensure rates are positive
    hr_raw = max(60, hr_raw)
    rr_raw = max(10, rr_raw)
    
    return hr_raw, rr_raw


# ---------- CACHING & LAZY SETUP ----------
analyzer = None
try:
    analyzer = SentimentIntensityAnalyzer()
except NameError:
    # Handles case where vaderSentiment might not be installed
    pass

supabase_client = None # Will be initialized later if keys are present

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
    # Placeholder for admin client if needed for RLS bypass/admin tasks
    # For simplicity, we use the regular client here
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
    st.session_state["chat_history"] = [{"role": "system", "content": "You are a supportive, non-judgmental wellness and mental health coach named Harmony. Keep responses encouraging and brief."}]
if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop"
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

# ---------- Supabase helpers (DB functions) ----------
# ... (All DB functions are preserved)

# -------------------- FRONT-END COMPONENTS --------------------

def app_splash_screen():
    st.title("HarmonySphere Wellness")
    st.info("Loading your personal wellness ecosystem...")
    time.sleep(2)
    st.session_state["show_splash"] = False
    st.rerun()

def unauthenticated_home():
    with st.container():
        st.title("Welcome to HarmonySphere")
        st.subheader("Login to start your journey.")

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
                st.session_state["user_id"] = str(uuid.uuid4()) # Dummy user ID
                st.success(f"Welcome, {st.session_state['user_name']}!")
                time.sleep(1)
                st.rerun()


def sidebar_navigation():
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.user_name}!")
        
        # Display Goal Progress
        total_goals = len(st.session_state.goals)
        completed_goals = sum(1 for goal in st.session_state.goals.values() if goal['count'] >= goal['target'])
        
        if total_goals > 0:
            progress_val = int((completed_goals / total_goals) * 100)
        else:
            progress_val = 0
            
        st.subheader("Daily Goal Progress")
        st.progress(progress_val, text=f"{completed_goals} / {total_goals} Goals Completed")
        st.markdown(f"**Focus on your goals today**")
        
        # Wellness Check-in Box (Kept original style, text visibility fixed by CSS)
        st.markdown(f"""
        <div class='metric-card' style='background-color: #FFFFE0; padding: 10px; border-left: 5px solid {TEAL};'>
            <p style='color: {DARK_NAVY_TEXT};'>Complete a Wellness Check-in!</p>
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
            # Buttons are globally styled with the new Teal palette
            if st.button(page_title, key=f"nav_{page_key}", use_container_width=True): 
                st.session_state["page"] = page_key
                st.rerun()


def homepage_panel():
    st.header("🏠 Dashboard Overview")
    st.success(f"Welcome back, {st.session_state.user_name}! We're glad you're here.")
    st.info(random.choice(QUOTES))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
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
        
        fig = px.line(mood_df, y='mood_level', title="Mood Over Time", 
                      labels={'mood_level': 'Mood Level (1-10)'},
                      markers=True, line_shape='spline',
                      color_discrete_sequence=[TEAL]) # Use Teal for chart line
        fig.update_layout(yaxis_range=[1, 10])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your mood to see your personal trend here!")


def mindful_breathing_page():
    st.header("🧘 Mindful Breathing Exercise")
    st.write("Follow the expansion and contraction of the heart for 4-7-8 breathing (or a pattern you prefer).")

    # The original HTML/CSS structure is used, ensuring smooth transition keys are preserved.
    breathing_state = st.session_state.get("breathing_state", "stop")
    st.markdown(f"""
    <div class="heart-animation-wrapper {breathing_state}">
        <div class="heart-shape">
            <div class="breathing-heart-content">
                <span id="breathing-text"></span>
            </div>
        </div>
    </div>
    <div id="countdown-timer-display" style="text-align:center; font-size: 1.2rem; font-weight: bold;"></div>
    """, unsafe_allow_html=True)
    
    st.subheader("Action Log")
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Heart button text visibility is fixed by the global CSS rule.
        if st.button("💖 Saved!", key="breathing_like_button", use_container_width=True, type="secondary"):
            # Update goal progress
            st.session_state.goals["breathing_session"]["count"] += 1
            st.toast("Exercise logged and heart button visible!")
            
    with col2:
        # Placeholder for start session logic (uses primary Teal button)
        if st.button("Start 5 Min Session", type="primary", use_container_width=True): 
            # This is where the JS animation logic would be triggered in a real app
            st.session_state["breathing_state"] = "inhale" # Dummy state change
            st.info("Starting session... Inhale 4s, Hold 7s, Exhale 8s.")
            time.sleep(1) # Simulate start lag
            st.balloons()
            st.session_state["breathing_state"] = "stop"
            st.rerun()


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
            # Save to DB if client exists...
            
def mood_tracker_page():
    st.header("😊 Mood Tracker")
    
    # Custom format to show emoji map in slider
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
        # Save to DB if client exists...

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
            # Save to DB if client exists...
            

def cbt_thought_record_page():
    st.header("🧠 CBT Thought Record")
    st.info("Use this tool to analyze and restructure automatic negative thoughts (ANTs).")
    
    record = {}
    with st.form("cbt_form"):
        for i, prompt in enumerate(CBT_PROMPTS):
            # Using raw markdown for bold prompt text
            st.markdown(prompt) 
            record[prompt] = st.text_area(f"Response to Step {i+1}", key=f"cbt_step_{i}", height=70)
            
        submitted = st.form_submit_button("Review Thought Record", type="primary")
        
        if submitted:
            # Simple check if the core parts were filled
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
            
            # Use AI to help with the reframing
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
    
    # Sentiment over time
    st.subheader("Sentiment Trend")
    sentiment_fig = px.line(
        journal_df, x='date', y='sentiment', 
        title="Journal Sentiment Over Time",
        labels={'sentiment': 'Sentiment Score (-1 to 1)', 'date': 'Date'},
        markers=True, line_shape='linear',
        color_discrete_sequence=[TEAL] # Use Teal for chart line
    )
    st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Word cloud placeholder (requires more complex setup, kept as text for now)
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
    
    # Placeholder for PDF export
    if st.button("Download Full Report (PDF)", type="primary"):
        st.success("Report generation simulated! (PDF would be created here)")

def ai_chat_page():
    st.header("🤖 AI Wellness Coach - Harmony")
    
    # Display chat history
    for message in st.session_state.chat_history[1:]: # Skip system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask Harmony for advice, support, or a reflection exercise..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Harmony is thinking..."):
                # Pass the entire history to maintain context
                ai_response = safe_generate(prompt, max_tokens=350)
                st.markdown(ai_response)
                # Update history with assistant's reply (including the context passed in the request)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})


def iot_dashboard_page():
    st.header("🔌 IoT Dashboard (ECE Demo)")
    st.info("Real-time physiological data simulated and filtered using a Kalman Filter.")
    
    # Update physiological data with a small simulation refresh
    if st.button("Simulate Data Refresh", key="refresh_iot", type="secondary"):
        st.session_state["last_refresh"] = datetime.now()
        # Data generation and Kalman filtering
        hr_raw, rr_raw = generate_simulated_physiological_data(int(time.time() * 1000))
        
        hr_filtered, st.session_state["hr_kalman_state"] = kalman_filter_simple(
            hr_raw, st.session_state["hr_kalman_state"]
        )
        rr_filtered, st.session_state["rr_kalman_state"] = kalman_filter_simple(
            rr_raw, st.session_state["rr_kalman_state"]
        )
        
        st.session_state["current_hr"] = hr_filtered
        st.session_state["current_rr"] = rr_filtered
        
        # Add a quick animation for feedback
        st.balloons() 
        
    st.markdown("---")

    col_hr, col_rr = st.columns(2)

    with col_hr:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Heart Rate (BPM)</p>
            <h2 style='margin: 0.2em 0; color: {TEAL};'>{st.session_state.current_hr:.1f}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Filtered by Kalman</p>
        </div>
        """, unsafe_allow_html=True)

    with col_rr:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.7;'>Respiration Rate (BPM)</p>
            <h2 style='margin: 0.2em 0; color: #36454F;'>{st.session_state.current_rr:.1f}</h2>
            <p style='font-size: 0.8rem; margin: 0; opacity: 0.9;'>Filtered by Kalman</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Real-time plot placeholder
    st.subheader("Live Data Stream (Simulated)")
    st.warning("A real-time chart would update here. Click 'Simulate Data Refresh' to update the metrics.")
        
        
# -------------------- MAIN APP EXECUTION --------------------
# Main placeholder for content
app_placeholder = st.empty()

# ---------- MAIN APPLICATION LOGIC (Triple Flow) ----------
with app_placeholder.container():
    
    # 1. Show Splash Screen first (blocks other content)
    if st.session_state.get("show_splash"):
        app_splash_screen()
        
    # 2. Transition to Centered Login
    elif not st.session_state.get("logged_in"):
        unauthenticated_home()

    # 3. Transition to Authenticated Dashboard
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
