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
    from supabase import create_client, Client
except ImportError:
    def create_client(*args, **kwargs):
        return None
    Client = type('DummyClient', (object,), {}) # Define a dummy Client class

# ---------- CONSTANTS AND CONFIGURATION ----------
# Ensure environment variables are set in Streamlit Secrets or your environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" 

PHQ9_CRISIS_THRESHOLD = 15 # Severe depression threshold

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
    10: "🥳 Joyful", 11: "✨ Euphoric"
}

DEFAULT_GOALS = {
    "journal_entry": {"name": "Write a Journal Entry", "count": 0, "target": 1, "emoji": "✍️"},
    "mood_log": {"name": "Log Your Mood", "count": 0, "target": 1, "emoji": "📊"},
    "breathing_session": {"name": "Mindful Breathing", "count": 0, "target": 1, "emoji": "🧘"},
}

# Wellness Ecosystem constants
ECO_PLANT_STATES = {
    "great": {"emoji": "🌸", "color": "#28a745", "message": "Your plant is thriving! Keep up the excellent work with your wellness routine."},
    "good": {"emoji": "🌱", "color": "#ffc107", "message": "Your plant is growing well, but a little more self-care could make it flourish!"},
    "poor": {"emoji": "💧", "color": "#dc3545", "message": "Your plant needs water! A little self-care today will help it recover."},
}

# Kalman Filter for IoT Dashboard (State)
if 'kalman_state' not in st.session_state:
    st.session_state['kalman_state'] = {
        'P': 1.0,  # Error covariance
        'X': 25.0, # Initial temperature estimate
    }

# ---------- Streamlit page config and CRITICAL STYLE INJECTION ----------
st.set_page_config(
    page_title="HarmonySphere", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CRITICAL STYLE INJECTION FIX ---
# This block defines the modern theme, card styles, and transitions.
st.markdown("""
<style>
/* 1. Global Background and Typography */
.stApp { 
    background: linear-gradient(135deg, #fcefee, #e0f7fa); /* soft pastel gradient */
    color: #1E1E1E; 
    font-family: 'Poppins', sans-serif; 
}
.main .block-container { 
    padding: 2rem 3rem;
}

/* 2. Streamlit Inputs (Text Area, Input Fields) */
textarea, input[type="text"], input[type="email"], input[type="password"] {
    color: #1E1E1E !important;
    -webkit-text-fill-color: #1E1E1E !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border: 2px solid #FFD6E0 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}
textarea:focus, input:focus {
    border-color: #FF9CC2 !important;
    box-shadow: 0 0 8px rgba(255, 156, 194, 0.5);
}

/* 3. Custom Card Style (Glassy / Wellness Look) */
.metric-card {
    padding: 25px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: transform 0.3s, box-shadow 0.3s, background 0.3s;
    margin-bottom: 20px;
    border: none;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.1);
    cursor: pointer;
    background: rgba(255, 255, 255, 0.9);
}

/* 4. Custom Sidebar Colors/Style (Calm Blue-Purple) */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #fff0f5, #e0f7fa); /* Light pastel gradient */
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
}

/* Sidebar Navigation Links/Buttons */
[data-testid="stSidebarNav"] li > a {
    color: #343a40; /* Darker text for better contrast */
    font-size: 1.0rem;
    border-radius: 8px;
    margin: 5px 0;
    padding: 10px 15px;
    transition: all 0.2s ease-in-out;
}

/* Sidebar Navigation Hover/Active */
[data-testid="stSidebarNav"] li > a:hover {
    background-color: #FFD6E0; /* Light pink on hover */
    color: #FF6F91;
    transform: translateX(5px);
}

/* Sidebar Active Page */
[data-testid="stSidebarNav"] li > a[aria-current="page"] {
    background-color: #FF9CC2; /* Primary color background for active page */
    color: white; 
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(255, 156, 194, 0.3);
}

/* 5. Primary Button Style (Pastel Rounded) */
.stButton>button {
    color: #FFFFFF;
    background: #FF9CC2;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s;
}
.stButton>button:hover {
    background: #FF6F91;
}

/* 6. Splash Screen Styling (for smooth transition) */
#splash-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: #FF9CC2; 
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    color: white;
    font-family: 'Poppins', sans-serif;
    transition: opacity 1s ease-out;
    opacity: 1;
}
.fade-out {
    opacity: 0 !important;
}
.spinner {
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top: 4px solid white;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin-top: 20px;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 7. Breathing Circle (Animated Calm Effect) */
.breathing-circle {
    width: 120px;
    height: 120px;
    background: #FF9CC2;
    border-radius: 50%;
    margin: 50px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    animation: scaleIn 4s infinite alternate ease-in-out; /* Default animation */
}
/* Re-define animations to be part of the style block */
@keyframes scaleIn {
    from { transform: scale(1); }
    to { transform: scale(1.5); }
}

</style>
""", unsafe_allow_html=True)
# --- END CRITICAL STYLE INJECTION FIX ---


# ---------- SUPABASE CLIENT FUNCTIONS ----------
@st.cache_resource
def get_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            st.error("Error creating Supabase client. Check SUPABASE_URL and SUPABASE_KEY.")
            return None
    return None

@st.cache_resource
def get_supabase_admin_client():
    """Gets the Supabase client using the Service Role Key (bypasses RLS)."""
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        except Exception:
            st.error("Error creating Supabase admin client. Check SUPABASE_SERVICE_ROLE_KEY.")
            return None
    return None

supabase_client = get_supabase_client()
supabase_admin_client = get_supabase_admin_client()

# ---------- UTILITY FUNCTIONS ----------
def get_openai_client():
    """Returns the configured OpenAI (OpenRouter) client."""
    if OPENROUTER_API_KEY:
        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )
    return None

def register_user_db(user_id, email, full_name):
    """Registers the user in the 'users' and 'profiles' tables using the Admin Client."""
    if not supabase_admin_client:
        return False, "Database not available for registration."

    try:
        # 1. Insert into 'users' table (required for RLS/security structure)
        supabase_admin_client.table('users').insert({"id": user_id, "email": email}).execute()
        
        # 2. Insert into 'profiles' table
        supabase_admin_client.table('profiles').insert({
            "user_id": user_id, 
            "full_name": full_name,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return True, "User registered successfully."
    except Exception as e:
        st.error(f"Database registration error: {e}")
        return False, f"Failed to register user in DB: {e}"

def load_all_user_data(user_id):
    """Fetches all user data (journal, mood, profile) and stores it in session state."""
    if not supabase_client:
        return False

    try:
        # Fetch Journal Entries
        journal_data = supabase_client.table('journal_entries').select('*').eq('user_id', user_id).order('date', desc=True).limit(100).execute()
        st.session_state["daily_journal"] = journal_data.data if journal_data.data else []
        
        # Fetch Mood History
        mood_data = supabase_client.table('mood_history').select('*').eq('user_id', user_id).order('date', desc=True).limit(100).execute()
        st.session_state["mood_history"] = mood_data.data if mood_data.data else []

        # Fetch Wellness Check-in (PHQ-9)
        phq9_data = supabase_client.table('wellness_checkin').select('*').eq('user_id', user_id).order('date', desc=True).limit(1).execute()
        if phq9_data.data:
            st.session_state["phq9_score"] = phq9_data.data[0]['score']
            st.session_state["phq9_data"] = phq9_data.data[0]
        else:
            st.session_state["phq9_score"] = None
            st.session_state["phq9_data"] = None

        # Fetch Profile Data (for full name)
        profile_data = supabase_client.table('profiles').select('full_name').eq('user_id', user_id).single().execute()
        st.session_state["user_full_name"] = profile_data.data['full_name']

        # Initialize Daily Goals (if not set)
        if "daily_goals" not in st.session_state or st.session_state.daily_goals is None:
            st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
            
        # Initialize AI chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello! I'm your AI wellness buddy. How can I support you today?"}
            ]
            
        # Initialize page navigation
        if "page" not in st.session_state:
            st.session_state["page"] = "Home"

        return True
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        st.session_state["logged_in"] = False
        return False

def check_and_reset_goals():
    """Resets daily goals if the date has changed since the last app load."""
    today_str = datetime.now().date().isoformat()
    
    # Defensive check for session state integrity
    if st.session_state.get("daily_goals") is None:
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
        st.session_state["last_goal_check"] = today_str
        return

    if st.session_state.get("last_goal_check") != today_str:
        # Reset goals
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
        st.session_state["last_goal_check"] = today_str
        st.sidebar.success("🎉 Daily goals reset for today!")

def update_goal(goal_key):
    """Increments a specific goal counter."""
    if goal_key in st.session_state.get("daily_goals", {}):
        st.session_state["daily_goals"][goal_key]["count"] += 1

def kalman_filter_simple(measured_value, process_noise=0.1, measurement_noise=0.3):
    """
    A simple 1D Kalman Filter for sensor data smoothing (used in IoT Dashboard).
    If kalman_state is somehow None, re-initialize it.
    """
    if st.session_state.get('kalman_state') is None:
        st.session_state['kalman_state'] = {'P': 1.0, 'X': measured_value}
        
    K = st.session_state['kalman_state'] # Current state

    # Prediction step
    P_predicted = K['P'] + process_noise # P = P + Q
    X_predicted = K['X']                # X = X 

    # Update step
    Kalman_Gain = P_predicted / (P_predicted + measurement_noise) # K = P' / (P' + R)
    X_updated = X_predicted + Kalman_Gain * (measured_value - X_predicted) # X = X' + K * (Z - X')
    P_updated = (1 - Kalman_Gain) * P_predicted # P = (1 - K) * P'

    # Save and return new state
    st.session_state['kalman_state'] = {
        'P': P_updated,
        'X': X_updated,
    }
    return X_updated

# ---------- AUTHENTICATION FUNCTIONS ----------

def sidebar_auth():
    """Handles login, registration, and logout in the sidebar."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    with st.sidebar:
        st.markdown(f"<h2>HarmonySphere 🧠</h2>", unsafe_allow_html=True)

        if not supabase_client:
             st.warning("⚠️ Supabase connection failed. App running in offline/demo mode.")

        if not st.session_state["logged_in"]:
            st.subheader("Welcome!")
            tab1, tab2 = st.tabs(["Log In", "Register"])
            
            with tab1:
                with st.form("login_form", clear_on_submit=False):
                    email = st.text_input("Email (Login)", key="login_email_input_fix")
                    password = st.text_input("Password (Login)", type="password", key="login_password_input_fix")
                    submit_button = st.form_submit_button("Login to Dashboard")

                    if submit_button:
                        if not supabase_client:
                            st.error("Cannot log in: Supabase client is not available.")
                            st.session_state["logged_in"] = False
                            st.session_state["user_id"] = None
                            st.stop()
                            
                        with st.spinner("Authenticating..."):
                            try:
                                response = supabase_client.auth.sign_in_with_password({'email': email, 'password': password})
                                
                                st.session_state["user"] = response.user
                                st.session_state["user_id"] = response.user.id
                                
                                # Load all data before setting logged_in
                                if load_all_user_data(response.user.id):
                                    st.session_state["logged_in"] = True
                                    st.sidebar.info("👋 Logged in successfully! Redirecting...")
                                    time.sleep(1.5) # Smooth transition delay
                                    st.rerun()
                                else:
                                    st.error("Login successful, but failed to load user data.")
                                    st.session_state["logged_in"] = False
                                    
                            except Exception as e:
                                st.error(f"Login failed: Invalid credentials or {e}")
                                st.session_state["logged_in"] = False

            with tab2:
                with st.form("register_form", clear_on_submit=True):
                    full_name = st.text_input("Full Name (Register)", key="register_name")
                    email = st.text_input("Email (Register)", key="register_email")
                    password = st.text_input("Password (min 6 characters)", type="password", key="register_password")
                    submit_button = st.form_submit_button("Create Account")

                    if submit_button:
                        if not supabase_admin_client:
                            st.error("Cannot register: Supabase admin client is not available.")
                            st.stop()

                        if len(password) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            with st.spinner("Creating account..."):
                                try:
                                    # Create user via Supabase Auth
                                    response = supabase_admin_client.auth.sign_up({'email': email, 'password': password})
                                    user_id = response.user.id
                                    
                                    # Register user data in our public tables using Admin Client
                                    success, message = register_user_db(user_id, email, full_name)
                                    
                                    if success:
                                        st.success("Account created! Please log in above.")
                                    else:
                                        # If DB registration fails, attempt to delete the auth user
                                        try:
                                            supabase_admin_client.auth.admin.delete_user(user_id)
                                        except:
                                            pass
                                        st.error(f"Registration failed: {message}")
                                        
                                except Exception as e:
                                    if "Email rate limit exceeded" in str(e):
                                        st.error("Too many signups from this IP. Please wait a moment.")
                                    else:
                                        st.error(f"Registration failed: {e}")

        else: # Logged In View
            st.markdown(f"#### Hello, {st.session_state.get('user_full_name', 'User')}!")
            st.caption(f"Logged in as: {st.session_state.get('user', {}).get('email', 'N/A')}")
            
            # --- DAILY GOALS TRACKER ---
            st.markdown("---")
            st.markdown("##### Today's Focus")
            goals = st.session_state.get("daily_goals", DEFAULT_GOALS)
            
            for key, goal in goals.items():
                progress = goal["count"] / goal["target"]
                emoji = goal["emoji"]
                
                if progress >= 1:
                    st.markdown(f"✅ **{emoji} {goal['name']}**")
                else:
                    st.markdown(f"**{emoji} {goal['name']}**")
                    st.progress(progress)
            
            st.markdown("---")
            
            if st.button("Logout", key="logout_button"):
                if supabase_client:
                    supabase_client.auth.sign_out()
                
                # Clear all session state variables
                for key in list(st.session_state.keys()):
                    if key not in ["app_loaded"]: # Keep splash screen state
                        del st.session_state[key]
                        
                st.session_state["logged_in"] = False
                st.session_state["page"] = "Home" # Reset page to avoid page error
                st.rerun()

# ---------- PAGE FUNCTIONS ----------

# --- 1. Homepage (MODERNIZED CARD STRUCTURE) --- 
def homepage_panel():
    
    # --- ULTRA SAFE DATA ACCESS FIX (Put this at the very top of the function) ---
    # Use the OR operator to default to an empty list [] if the value is None
    safe_journal = st.session_state.get("daily_journal") or []
    safe_mood = st.session_state.get("mood_history") or []

    # Now, calculate metrics using the safe variables
    total_entries = len(safe_journal)
    df_mood = pd.DataFrame(safe_mood)
    # ------------------------------------------------------------
    
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #FF6F91;'>🧠</span></h1>", unsafe_allow_html=True)
    st.caption("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")

    # --- CRISIS ALERT ---
    if st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] >= PHQ9_CRISIS_THRESHOLD:
        st.error("🚨 **CRISIS ALERT:** Your last Wellness Check-in indicated a high level of distress. Please prioritize contacting a helpline or trusted adult immediately. Your safety is paramount.")
    
    st.markdown("---")

    # Calculate metrics for cards
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else 6.0 
    
    # Calculate a simple streak
    current_streak = 0
    if st.session_state.get("mood_history"):
        # Convert all mood history dates to date objects
        dates = pd.to_datetime([item['date'] for item in st.session_state['mood_history']]).date
        
        # Get unique dates
        unique_dates = set(dates)
        today = datetime.now().date()
        
        # Check for streak starting today
        current_date = today
        while current_date in unique_dates:
            current_streak += 1
            current_date -= timedelta(days=1)
        
    
    # --- METRIC CARDS ---
    st.subheader("Quick Glance")
    col1, col2, col3 = st.columns(3)

    # --- Card 1: Total Journal Entries ---
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5em; font-weight: 700; color: #FF9CC2;">{total_entries}</div>
            <div style="font-size: 1.0em; color: #6c757d;">Total Journal Entries</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Card 2: Average Sentiment Score ---
    with col2:
        sentiment_color = "#28a745" if avg_mood_7d >= 6.5 else "#ffc107" if avg_mood_7d >= 5.5 else "#dc3545"
        sentiment_label = MOOD_EMOJI_MAP.get(int(round(avg_mood_7d)), "❓")

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5em; font-weight: 700; color: {sentiment_color};">{sentiment_label}</div>
            <div style="font-size: 1.0em; color: #6c757d;">7-Day Avg Mood ({avg_mood_7d:.1f})</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Card 3: Current Streak ---
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5em; font-weight: 700; color: #007bff;">{current_streak} Days 🔥</div>
            <div style="font-size: 1.0em; color: #6c757d;">Current Streak</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Your Daily Focus ✨")

    # Calculate relevant metrics for nudge
    avg_mood_7d_nudge = df_mood.head(7)['mood'].mean() if not df_mood.empty else None

    if avg_mood_7d_nudge is None:
        avg_mood_display = "N/A"
        mood_icon = "❓"
    else:
        avg_mood_display = f"{avg_mood_7d_nudge:.1f}"
        mood_icon = MOOD_EMOJI_MAP.get(int(round(avg_mood_7d_nudge)), "❓")
        
    col_nudge, col_quote = st.columns([3, 1])

    with col_nudge:
        with st.container(border=True):
            if st.session_state.get("daily_goals", {}).get("journal_entry", {}).get("count", 0) < 1:
                st.info("💡 **Daily Goal:** Haven't journaled today? Take 5 minutes for a quick 'brain dump' on the **Mindful Journaling** page to clear your mind.")
            elif avg_mood_7d_nudge is not None and avg_mood_7d_nudge <= 5.0:
                 st.warning(f"⚠️ **Check-in Alert:** Your 7-day average mood is trending low ({avg_mood_display} {mood_icon}). Consider a **CBT Thought Record** session to reframe a difficult thought, or reach out to the **AI Chat** buddy.")
            elif st.session_state.get("phq9_score") is None:
                st.info("📊 **First Step:** Take the quick **Wellness Check-in** on the sidebar. It helps us personalize your journey!")
            else:
                st.success(f"🌟 **Great Work!** Your average mood is {avg_mood_display} {mood_icon}. Check the **Wellness Ecosystem** for your plant's status.")

    with col_quote:
        quote = random.choice(QUOTES)
        st.info(f"**Inspiration:**\n\n>{quote}")

    st.subheader("Your Progress Over Time")
    
    # --- MOOD CHART ---
    if not df_mood.empty:
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        df_mood = df_mood.sort_values(by='date').reset_index(drop=True)
        df_mood['7-Day Avg'] = df_mood['mood'].rolling(window=7, min_periods=1).mean()
        
        fig = px.line(
            df_mood, 
            x="date", 
            y="7-Day Avg", 
            title="Mood Trend (7-Day Rolling Average)",
            labels={"date": "Date", "7-Day Avg": "Average Mood Score (1-11)"},
            template="plotly_white"
        )
        fig.update_traces(line_color='#FF9CC2', line_width=3)
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f7f9fb",
            font=dict(family="Poppins")
        )

        with st.container(border=True):
             st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mood data yet. Log your first mood to see your trends!")

# --- 2. Mindful Journaling ---
def mindful_journaling_page():
    st.title("Mindful Journaling ✍️")
    st.caption("A space for free-form reflection. Simply write what's on your mind.")

    with st.form("journal_form", clear_on_submit=True):
        entry_date = st.date_input("Date of Entry", datetime.now().date(), key="journal_date")
        title = st.text_input("Title (Optional)", key="journal_title")
        content = st.text_area("What's on your mind today?", height=300, key="journal_content")
        submit_button = st.form_submit_button("Save Entry")

        if submit_button and content:
            if not supabase_client:
                st.error("Cannot save entry: Supabase client is not available.")
                st.stop()
                
            analyzer = SentimentIntensityAnalyzer()
            sentiment_score = analyzer.polarity_scores(content)['compound']
            
            try:
                supabase_client.table('journal_entries').insert({
                    "user_id": st.session_state["user_id"],
                    "date": entry_date.isoformat(),
                    "title": title,
                    "content": content,
                    "sentiment_score": sentiment_score
                }).execute()

                st.success("Journal entry saved and analyzed!")
                update_goal("journal_entry")
                
                # Fetch new data and rerun to update homepage/sidebar
                load_all_user_data(st.session_state["user_id"]) 
                st.rerun() 
            except Exception as e:
                st.error(f"Failed to save journal entry: {e}")

    st.subheader("Recent Entries")
    df_journal = pd.DataFrame(st.session_state.get("daily_journal", []))
    if not df_journal.empty:
        df_journal['date'] = pd.to_datetime(df_journal['date']).dt.date
        
        # Display as expandable elements
        for index, row in df_journal.head(5).iterrows():
            with st.expander(f"**{row['date']}** - {row['title'] if row['title'] else 'Untitled Entry'}"):
                st.markdown(f"**Sentiment Score:** {row['sentiment_score']:.2f}")
                st.markdown(row['content'])
    else:
        st.info("You haven't saved any journal entries yet.")

# --- 3. Mood Tracker ---
def mood_tracker_page():
    st.title("Mood Tracker 📊")
    st.caption("Rate your general mood to track your emotional trends.")
    
    with st.form("mood_form", clear_on_submit=True):
        mood_date = st.date_input("Date of Mood Log", datetime.now().date(), key="mood_date")
        
        mood_score = st.slider(
            "How would you rate your mood today (1 being the lowest, 11 being the highest)?",
            min_value=1,
            max_value=11,
            value=6,
            step=1
        )
        st.markdown(f"Your selected mood: **{MOOD_EMOJI_MAP.get(mood_score)}**")
        
        notes = st.text_area("Any specific feelings or events?", key="mood_notes")
        submit_button = st.form_submit_button("Save Mood")

        if submit_button:
            if not supabase_client:
                st.error("Cannot save mood: Supabase client is not available.")
                st.stop()
                
            try:
                supabase_client.table('mood_history').insert({
                    "user_id": st.session_state["user_id"],
                    "date": mood_date.isoformat(),
                    "mood": mood_score,
                    "notes": notes
                }).execute()

                st.success("Mood successfully logged!")
                update_goal("mood_log")
                
                # Fetch new data and rerun to update homepage/sidebar
                load_all_user_data(st.session_state["user_id"])
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save mood log: {e}")

    st.subheader("Your Mood History")
    df_mood = pd.DataFrame(st.session_state.get("mood_history", []))
    
    if not df_mood.empty:
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        
        # Calculate weekly average
        df_mood['Week'] = df_mood['date'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
        df_weekly_avg = df_mood.groupby('Week')['mood'].mean().reset_index()
        df_weekly_avg.rename(columns={'mood': 'Weekly Average Mood'}, inplace=True)
        
        fig = px.bar(
            df_weekly_avg.sort_values('Week', ascending=False).head(8).sort_values('Week'),
            x='Week',
            y='Weekly Average Mood',
            title='Last 8 Weeks Average Mood',
            labels={'Week': 'Start of Week', 'Weekly Average Mood': 'Avg Mood Score (1-11)'},
            color_discrete_sequence=['#FF9CC2']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start logging your mood to see your history chart here.")

# --- 4. Wellness Check-in (PHQ-9 Simulator) ---
def wellness_checkin_page():
    st.title("Wellness Check-in (PHQ-9) 📋")
    st.caption("The Patient Health Questionnaire (PHQ-9) is a quick self-assessment. Be honest with your answers, your well-being matters.")
    
    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself—or that you are a failure or have let yourself or your family down?",
        "Trouble concentrating on things, such as reading the newspaper or watching television?",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?",
        "Thoughts that you would be better off dead, or of hurting yourself in some way?"
    ]
    
    response_map = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    
    with st.form("phq9_form"):
        st.markdown("Over the last 2 weeks, how often have you been bothered by any of the following problems?")
        
        scores = []
        for i, q in enumerate(questions):
            response = st.radio(f"**Q{i+1}:** {q}", list(response_map.keys()), key=f"phq9_{i}")
            scores.append(response_map[response])
            
        st.markdown("---")
        submit_button = st.form_submit_button("Calculate Score")
        
        if submit_button:
            total_score = sum(scores)
            
            # Save to DB
            if supabase_client:
                try:
                    supabase_client.table('wellness_checkin').insert({
                        "user_id": st.session_state["user_id"],
                        "date": datetime.now().date().isoformat(),
                        "score": total_score,
                        "scores_array": scores
                    }).execute()
                    
                    st.session_state["phq9_score"] = total_score
                    load_all_user_data(st.session_state["user_id"])
                except Exception as e:
                    st.error(f"Failed to save check-in: {e}")
            
            st.subheader(f"Your PHQ-9 Score: {total_score}")
            
            # Interpretation
            if total_score <= 4:
                st.success("Minimal Depression. Maintain your current wellness habits.")
            elif 5 <= total_score <= 9:
                st.info("Mild Depression. Monitor your mood. Focus on self-care and journaling.")
            elif 10 <= total_score <= 14:
                st.warning("Moderate Depression. Consider speaking with a trusted professional or using the CBT tool.")
            elif 15 <= total_score <= 19:
                st.error("Moderately Severe Depression. It is important to seek professional help.")
            elif total_score >= 20:
                st.markdown("## 🚨 Severe Depression / Crisis Alert 🚨")
                st.markdown("Your safety is paramount. **Please contact a professional or a crisis line immediately.**")
                st.error("Crisis Line Placeholder: [1-800-273-8255 (US Suicide & Crisis Lifeline)]")

# --- 5. AI Chat (OpenRouter/OpenAI) ---
def ai_chat_page():
    st.title("AI Wellness Buddy 🤖")
    st.caption("Your non-judgmental partner for a quick chat, motivational words, or stress relief ideas.")
    
    openai_client = get_openai_client()
    if not openai_client:
        st.warning("AI Chat is offline. Please set the OPENROUTER_API_KEY environment variable.")
        return

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your wellness buddy anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI is thinking..."):
                try:
                    # System prompt for context
                    system_prompt = (
                        "You are a compassionate and non-judgmental AI wellness buddy called HarmonySphere. "
                        "Your tone should be gentle, encouraging, and supportive. "
                        "Provide mindfulness tips, reframing exercises, and general emotional support. "
                        "You are NOT a substitute for professional mental health care. Always prioritize safety."
                    )
                    
                    # Combine system prompt with history
                    full_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                    
                    stream = openai_client.chat.completions.create(
                        model=OPENROUTER_MODEL_NAME,
                        messages=full_messages,
                        stream=True,
                    )
                    
                    response = st.write_stream(stream)
                    
                except APIError as e:
                    response = f"An API error occurred: {e}"
                    st.error(response)
                except Exception as e:
                    response = f"An unexpected error occurred: {e}"
                    st.error(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- 6. Wellness Ecosystem ---
def wellness_ecosystem_page():
    st.title("Wellness Ecosystem 🌱")
    st.caption("Your digital plant reflects your consistency in self-care. Keep it thriving!")
    
    goals = st.session_state.get("daily_goals", DEFAULT_GOALS)
    
    # Calculate Wellness Score: 1 point for meeting any daily goal
    wellness_score = sum(1 for goal in goals.values() if goal["count"] >= goal["target"])
    
    # Calculate overall consistency (Example logic: 7-day mood average)
    df_mood = pd.DataFrame(st.session_state.get("mood_history") or [])
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else 6.0 
    
    # Determine Plant Health based on mood trend and daily goals
    if wellness_score >= 2 and avg_mood_7d >= 7.0:
        health_state = "great"
    elif wellness_score >= 1 or avg_mood_7d >= 5.5:
        health_state = "good"
    else:
        health_state = "poor"
        
    plant = ECO_PLANT_STATES[health_state]
    
    # --- Visual Plant Display ---
    with st.container(border=True):
        col_plant, col_message = st.columns([1, 2])
        
        with col_plant:
            st.markdown(f"""
            <div style="text-align: center; font-size: 5rem; line-height: 1;">{plant['emoji']}</div>
            <div style="text-align: center; font-size: 1.5rem; color: {plant['color']}; font-weight: 700;">{health_state.capitalize()} Health</div>
            """, unsafe_allow_html=True)

        with col_message:
            st.subheader(f"Your Plant Status: {plant['emoji']}")
            st.info(plant['message'])
            
            st.markdown("---")
            st.markdown("##### Plant Health Summary:")
            st.markdown(f"- **Daily Goals Achieved:** **{wellness_score}/3**")
            st.markdown(f"- **7-Day Mood Average:** **{avg_mood_7d:.1f}/11**")

# --- 7. Mindful Breathing ---
def mindful_breathing_page():
    st.title("Mindful Breathing 🧘")
    st.caption("A simple 4-7-8 breathing technique to calm your nervous system.")

    st.warning("Click the button to begin the exercise. This page uses a visual indicator, but follow the counts for best results.")

    if st.button("Start 4-7-8 Breathing Cycle", key="start_breathing"):
        st.empty() # Clear button after click
        st.session_state["breathing_active"] = True
        update_goal("breathing_session")
        
        st.sidebar.success("✅ Goal Achieved: Mindful Breathing")
        
    if st.session_state.get("breathing_active"):
        
        # Simple animation loop
        placeholder = st.empty()
        
        for _ in range(3): # Run 3 cycles
            # INHALE (4 seconds)
            with placeholder.container():
                st.markdown('<div class="breathing-circle breathe-inhale" style="background-color: #4CAF50;">INHALE 4</div>', unsafe_allow_html=True)
            time.sleep(4)
            
            # HOLD (7 seconds)
            with placeholder.container():
                 st.markdown('<div class="breathing-circle" style="background-color: #FFC107; transform: scale(1.5);">HOLD 7</div>', unsafe_allow_html=True)
            time.sleep(7)
            
            # EXHALE (8 seconds)
            with placeholder.container():
                st.markdown('<div class="breathing-circle" style="background-color: #F44336; animation: scaleOut 8s infinite alternate ease-in-out;">EXHALE 8</div>', unsafe_allow_html=True)
            time.sleep(8)
            
        # End of session
        st.session_state["breathing_active"] = False
        placeholder.empty()
        st.success("Session complete! You can start another one or return to the main page.")
        # Reload sidebar to show goal update
        st.rerun()

# --- 8. CBT Thought Record ---
def cbt_thought_record_page():
    st.title("CBT Thought Record 🧠")
    st.caption("Use this to challenge negative automatic thoughts and practice cognitive restructuring.")
    
    with st.form("cbt_form", clear_on_submit=True):
        st.subheader("1. Situation & Thought")
        date_cbt = st.date_input("Date", datetime.now().date(), key="cbt_date")
        situation = st.text_area("What was the situation? (Who, what, where, when)", key="cbt_situation")
        thought = st.text_area("What was the Automatic Negative Thought (ANT)?", key="cbt_thought")

        st.subheader("2. Evidence & Rebuttal")
        evidence_for = st.text_area("What is the evidence that the ANT is **TRUE**?", key="cbt_evidence_for")
        evidence_against = st.text_area("What is the evidence that the ANT is **NOT TRUE**?", key="cbt_evidence_against")

        st.subheader("3. Balanced Thought")
        balanced_thought = st.text_area("What is a more realistic and balanced thought?", key="cbt_balanced")
        
        submit_button = st.form_submit_button("Save Record")

        if submit_button and balanced_thought:
            if not supabase_client:
                st.error("Cannot save record: Supabase client is not available.")
                st.stop()
                
            try:
                supabase_client.table('cbt_records').insert({
                    "user_id": st.session_state["user_id"],
                    "date": date_cbt.isoformat(),
                    "situation": situation,
                    "automatic_thought": thought,
                    "evidence_for": evidence_for,
                    "evidence_against": evidence_against,
                    "balanced_thought": balanced_thought
                }).execute()

                st.success("CBT Thought Record saved successfully!")
            except Exception as e:
                st.error(f"Failed to save CBT record: {e}")

# --- 9. Journal Analysis ---
def journal_analysis_page():
    st.title("Journal Analysis 🔍")
    st.caption("AI-powered insights into your recent journal entries.")
    
    openai_client = get_openai_client()
    if not openai_client:
        st.warning("Analysis is offline. Please set the OPENROUTER_API_KEY environment variable.")
        return

    df_journal = pd.DataFrame(st.session_state.get("daily_journal", [])).head(5)

    if df_journal.empty:
        st.info("Please write at least one journal entry to enable analysis.")
        return

    # Prepare context for AI
    journal_context = "\n---\n".join([
        f"Date: {row['date']}\nTitle: {row['title']}\nContent: {row['content']}\nSentiment: {row['sentiment_score']:.2f}"
        for _, row in df_journal.iterrows()
    ])
    
    analysis_prompt = (
        "Analyze the following recent journal entries for recurring themes, primary emotional tones (e.g., anxiety, joy, stress), "
        "and any potential triggers mentioned. Provide a summary of the user's focus over the last few days and one actionable, supportive insight."
        f"\n\n--- Journal Entries ---\n{journal_context}"
    )

    if st.button("Generate AI Insight", key="generate_insight"):
        with st.spinner("Analyzing your thoughts..."):
            try:
                response = openai_client.chat.completions.create(
                    model=OPENROUTER_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a specialized journal analyst, providing compassionate, insightful, and actionable feedback based on the provided text. Keep the tone gentle and non-clinical."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                )
                
                st.subheader("AI Analysis Summary")
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Analysis Failed: {e}")

# --- 10. IoT Dashboard (ECE Project Example) ---
def iot_dashboard_page():
    st.title("IoT Dashboard (ECE Project Example) ⚙️")
    st.caption("Simulated data from an environment sensor (e.g., for a student dorm/lab).")
    
    st.warning("This is a simulated dashboard to showcase data visualization and filtering.")

    # Generate simulated data
    current_time = datetime.now()
    data_points = 50
    time_series = [current_time - timedelta(minutes=i) for i in range(data_points)]
    
    # Simulate a noisy temperature signal around 24.5C
    sim_temp = 24.5 + np.random.normal(0, 1.5, data_points)
    
    # Apply Kalman Filter for smoothing (in reverse time order)
    smoothed_temp = []
    
    # Reset Kalman state for consistent simulation start
    if 'kalman_state' in st.session_state:
        del st.session_state['kalman_state']

    for temp in reversed(sim_temp):
        smoothed = kalman_filter_simple(temp)
        smoothed_temp.append(smoothed)
    
    df_iot = pd.DataFrame({
        'Time': reversed(time_series),
        'Measured Temperature (°C)': sim_temp,
        'Smoothed Temperature (°C)': smoothed_temp,
        'Humidity (%)': 50 + np.random.normal(0, 2, data_points) # Simulate humidity
    })
    
    # --- Visualization ---
    st.subheader("Temperature Trend")
    
    fig = px.line(
        df_iot, 
        x='Time', 
        y=['Measured Temperature (°C)', 'Smoothed Temperature (°C)'],
        title='Environment Temperature Over Time',
        color_discrete_map={'Measured Temperature (°C)': '#FFC107', 'Smoothed Temperature (°C)': '#007bff'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Current Readings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Temp (Smoothed)", f"{df_iot['Smoothed Temperature (°C)'].iloc[-1]:.2f} °C", delta=f"{df_iot['Smoothed Temperature (°C)'].iloc[-1] - df_iot['Smoothed Temperature (°C)'].iloc[-2]:.2f} °C")
    with col2:
        st.metric("Current Humidity", f"{df_iot['Humidity (%)'].iloc[-1]:.2f} %")
    with col3:
        # Example of a derived metric
        comfort_index = (df_iot['Smoothed Temperature (°C)'].iloc[-1] + df_iot['Humidity (%)'].iloc[-1] / 100) / 2
        st.metric("Comfort Index", f"{comfort_index:.2f}")

# --- 11. Report & Summary ---
def report_summary_page():
    st.title("Report & Summary 📈")
    st.caption("A consolidated view of your wellness metrics.")

    df_mood = pd.DataFrame(st.session_state.get("mood_history") or [])
    df_journal = pd.DataFrame(st.session_state.get("daily_journal") or [])
    
    st.subheader("Key Wellness Indicators")
    col1, col2, col3 = st.columns(3)
    
    # Mood Indicator
    avg_mood = df_mood['mood'].mean() if not df_mood.empty else 6.0
    mood_emoji = MOOD_EMOJI_MAP.get(int(round(avg_mood)), "❓")
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #007bff;">
            <div style="font-size: 1.5em; color: #6c757d;">Overall Avg Mood</div>
            <div style="font-size: 2.5em; font-weight: 700; color: #007bff;">{avg_mood:.1f} {mood_emoji}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sentiment Indicator
    avg_sentiment = df_journal['sentiment_score'].mean() if not df_journal.empty else 0.0
    sentiment_color = "#28a745" if avg_sentiment >= 0.2 else "#dc3545" if avg_sentiment <= -0.1 else "#ffc107"
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {sentiment_color};">
            <div style="font-size: 1.5em; color: #6c757d;">Journal Sentiment</div>
            <div style="font-size: 2.5em; font-weight: 700; color: {sentiment_color};">{avg_sentiment:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # PHQ-9 Score
    phq9_score = st.session_state.get("phq9_score")
    phq9_color = "#dc3545" if phq9_score is not None and phq9_score >= 10 else "#28a745" if phq9_score is not None and phq9_score <= 4 else "#ffc107"
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {phq9_color};">
            <div style="font-size: 1.5em; color: #6c757d;">Last PHQ-9 Score</div>
            <div style="font-size: 2.5em; font-weight: 700; color: {phq9_color};">{phq9_score if phq9_score is not None else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Journal Sentiment Trend")
    if not df_journal.empty:
        df_journal['date'] = pd.to_datetime(df_journal['date'])
        fig = px.scatter(
            df_journal,
            x='date',
            y='sentiment_score',
            title='Sentiment of Journal Entries Over Time',
            color_discrete_sequence=['#FF9CC2'],
            trendline='lowess'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log a few journal entries to see your sentiment trend.")

# --- Unauthenticated Home ---
def unauthenticated_home():
    st.title("Welcome to HarmonySphere 🌸")
    st.markdown("Your personal, safe space for mental wellness and self-care.")

    with st.container(border=True):
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #fcefee;">
            <h3 style="color: #FF6F91; margin-top: 0;">Access Your Dashboard</h3>
            <p>Please use the login or register form on the **left sidebar** to access the app's features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Remember: HarmonySphere is a support tool, not a substitute for medical advice.")


# ---------- SPLASH SCREEN LOGIC (For smooth startup) ----------

def show_splash_screen():
    """Displays a splash screen with a loading animation."""
    st.markdown("""
        <div id="splash-container">
            <h1>HarmonySphere</h1>
            <p>Loading your personalized wellness journey...</p>
            <div class="spinner"></div>
        </div>
        <script>
        // Use JavaScript to fade out the splash screen after a delay
        setTimeout(function() {
            var splash = document.getElementById('splash-container');
            if (splash) {
                splash.classList.add('fade-out');
            }
        }, 1500); // Display for 1.5 seconds
        
        setTimeout(function() {
            var splash = document.getElementById('splash-container');
            if (splash) {
                splash.style.display = 'none'; // Completely hide after fade
            }
        }, 2500); // 1.5s display + 1s fade-out
        </script>
        """, unsafe_allow_html=True)
    
    # This must run before the main content to initialize state correctly
    st.session_state["app_loaded"] = True
    time.sleep(2.5) # Wait for fade-out to complete visually
    st.experimental_rerun()


# ---------- MAIN APPLICATION LOGIC ----------

# CRITICAL: Run the splash screen on first load only
if "app_loaded" not in st.session_state:
    show_splash_screen()

# CRITICAL: Always run sidebar auth logic
sidebar_auth()

if not st.session_state.get("logged_in"):
    unauthenticated_home()

else:
    # --- AUTHENTICATED PAGES ---
    # Perform daily goal check on every run while logged in
    check_and_reset_goals()
    
    current_page = st.session_state["page"]
    
    # 1. Fully Built Pages (Functional features)
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
    elif current_page == "Wellness Ecosystem":
        wellness_ecosystem_page()
    
    # 2. Pages with Simple Functionality
    elif current_page == "Mindful Breathing":
        mindful_breathing_page()
    elif current_page == "CBT Thought Record":
        cbt_thought_record_page()
    elif current_page == "Journal Analysis":
        journal_analysis_page()
    
    # 3. Project-Specific Pages
    elif current_page == "IoT Dashboard (ECE)": 
        iot_dashboard_page()
    elif current_page == "Report & Summary": 
        report_summary_page()
    
    else:
        st.error("Page not found.")