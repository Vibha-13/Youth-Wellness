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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI
from openai import APIError

# Placeholder for Supabase client (assuming database is used for persistence)
try:
    from supabase import create_client
except ImportError:
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
    10: "🥳 Euphoric"
}

MOOD_FACTORS = [
    "Work/School Stress", "Relationships", "Health (Physical)",
    "Sleep Quality", "Financial Concerns", "Weather",
    "Exercise", "Diet", "Current Events", "Relaxation Time"
]

# ---------- HELPER FUNCTIONS ----------

def set_page(page_name):
    """Utility to switch pages via session state."""
    st.session_state["page"] = page_name

def load_data(table_name):
    """Placeholder for loading user-specific data from Supabase."""
    # In a real app, this would query the DB
    return []

def save_data(table_name, data):
    """Placeholder for saving data to Supabase."""
    # In a real app, this would insert/upsert data
    st.toast(f"Mock: Saving to {table_name}: {data['mood_score']}...")
    time.sleep(0.5)
    return True

def get_db():
    """Initializes and returns the Supabase client."""
    if "db" not in st.session_state:
        # In a real application, replace these with actual secrets
        SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://mock.supabase.co")
        SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "mock-key")
        st.session_state["db"] = create_client(SUPABASE_URL, SUPABASE_KEY)
    return st.session_state.get("db")

def is_db_ready():
    """Checks if the database client is initialized."""
    return get_db() is not None

# ---------- PAGE IMPLEMENTATIONS (Placeholders Restored) ----------

def homepage_panel():
    st.header("Welcome Back! 👋")
    st.write(f"Today's affirmation: _{random.choice(QUOTES)}_")
    st.success("This is the main dashboard content.")
    
def mindful_journaling_page():
    st.header("Mindful Journaling ✍️")
    st.info("This is where you can write down your thoughts and feelings.")

# RESTORED: Placeholder
def wellness_checkin_page():
    st.header("Wellness Check-in 🧘")
    st.info("This page is for tracking holistic wellness metrics (e.g., nutrition, hydration).")

# RESTORED: Placeholder
def wellness_ecosystem_page():
    st.header("Wellness Ecosystem 🌐")
    st.info("This page will feature integrations with other wellness platforms and services.")

def ai_chat_page():
    st.header("AI Chat 🤖")
    st.info("Your AI companion for supportive conversation.")

def mindful_breathing_page():
    st.header("Mindful Breathing 🌬️")
    st.info("A guided breathing exercise to calm your nervous system.")

def cbt_thought_record_page():
    st.header("CBT Thought Record 🧠")
    st.info("Work through challenging thoughts using the CBT framework.")

# RESTORED: Placeholder
def journal_analysis_page():
    st.header("Journal Analysis 📊")
    st.info("This page will show sentiment and theme analysis of your journal entries.")

def report_summary_page():
    st.header("Report & Summary 📈")
    st.info("View your long-term mood and journaling analytics here.")

# Functioning IoT Dashboard
def iot_dashboard_page():
    st.title("Environmental Wellness Dashboard 📊")
    st.markdown("Monitor and visualize physiological and environmental metrics from integrated IoT devices.")
    st.markdown("---")

    # --- 1. Mock Data Generation ---
    # Create mock data for the last 24 hours
    now = datetime.now()
    time_series = [now - timedelta(hours=i) for i in range(24)]
    time_series.reverse() # Sort from oldest to newest

    # Mock data simulation with some realistic fluctuations
    temperature = np.random.normal(25, 1, 24) # Avg 25 C
    humidity = np.random.normal(55, 5, 24)    # Avg 55 %
    light_level = np.random.normal(400, 100, 24) # Avg 400 Lux

    df = pd.DataFrame({
        "Timestamp": time_series,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "Light Level (Lux)": light_level
    })
    
    # --- 2. Key Metrics ---
    st.subheader("Current Environment")
    col1, col2, col3 = st.columns(3)

    # Get the latest values
    latest_temp = round(df["Temperature (°C)"].iloc[-1], 1)
    latest_humidity = round(df["Humidity (%)"].iloc[-1], 1)
    latest_light = round(df["Light Level (Lux)"].iloc[-1], 0)

    col1.metric("Temperature", f"{latest_temp} °C", delta=f"{round(latest_temp - df['Temperature (°C)'].iloc[-2], 1)} °C")
    col2.metric("Humidity", f"{latest_humidity} %", delta=f"{round(latest_humidity - df['Humidity (%)'].iloc[-2], 1)} %")
    col3.metric("Light Level", f"{latest_light} Lux", delta=f"{round(latest_light - df['Light Level (Lux)'].iloc[-2], 0)}")
    
    st.markdown("---")

    # --- 3. Interactive Charts using Plotly ---
    st.subheader("Last 24 Hours Trends")

    # Temperature Chart
    fig_temp = px.line(df, x="Timestamp", y="Temperature (°C)", 
                       title="Temperature Variation", 
                       template="plotly_white",
                       color_discrete_sequence=["#FF7F0E"])
    fig_temp.update_traces(mode='lines+markers', marker_size=5)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Humidity Chart
    fig_humidity = px.line(df, x="Timestamp", y="Humidity (%)", 
                           title="Humidity Variation", 
                           template="plotly_white",
                           color_discrete_sequence=["#1F77B4"])
    fig_humidity.update_traces(mode='lines+markers', marker_size=5)
    st.plotly_chart(fig_humidity, use_container_width=True)

    # Light Level Chart
    fig_light = px.line(df, x="Timestamp", y="Light Level (Lux)", 
                        title="Light Level Variation", 
                        template="plotly_white",
                        color_discrete_sequence=["#2CA02C"])
    fig_light.update_traces(mode='lines+markers', marker_size=5)
    st.plotly_chart(fig_light, use_container_width=True)


# ---------- MOOD TRACKER PAGE (Modified) ----------

def mood_tracker_page():
    st.title("Mood Tracker 🗓️")
    st.markdown("Track your emotional well-being to identify patterns and understand what influences your feelings.")
    st.markdown("---")

    # 1. Mood Selection
    current_mood_score = st.slider(
        "How are you feeling right now? (1=Lowest, 10=Highest)",
        min_value=1,
        max_value=10,
        value=st.session_state.get('mood_score_input', 5), # Keep state in session
        step=1,
        format=f"%d {MOOD_EMOJI_MAP.get(st.session_state.get('mood_score_input', 5), '😐')}",
        key='mood_score_input',
        help="Select a score from 1 (Agonizing) to 10 (Euphoric)."
    )
    
    # Update the slider label dynamically immediately after interaction
    st.markdown(f"**Selected Mood:** **{MOOD_EMOJI_MAP.get(current_mood_score)}**")
    st.markdown("---")


    # 2. Factors Selection
    selected_factors = st.multiselect(
        "What factors are influencing your mood?",
        options=MOOD_FACTORS,
        default=st.session_state.get('mood_factors_input', []),
        key='mood_factors_input'
    )
    
    # 3. Reflection
    reflection_text = st.text_area(
        "Write a quick reflection or journal entry (optional)",
        value=st.session_state.get('reflection_input', ''),
        key='reflection_input'
    )
    
    # 4. Save Button
    if st.button("Save Mood Entry", type="primary", use_container_width=True):
        if not current_mood_score:
            st.error("Please select a mood score before saving.")
            return

        # Prepare the data dictionary
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "mood_score": current_mood_score,
            "mood_emoji": MOOD_EMOJI_MAP.get(current_mood_score),
            "factors": selected_factors,
            "reflection": reflection_text,
            "user_id": st.session_state.get("user_id", "guest")
        }

        # Mock save operation
        if is_db_ready():
            # In a real app, use the actual database call
            save_data("mood_entries", new_entry)
        
        st.success("Mood entry saved successfully!")
        
        # Clear the input fields after saving
        del st.session_state['mood_score_input']
        del st.session_state['mood_factors_input']
        del st.session_state['reflection_input']
        st.experimental_rerun() # Rerun to clear the fields

    st.markdown("---")

    # NEW: Conditional Supportive Intervention
    # This section appears only when a low mood is selected (4 or below)
    if current_mood_score <= 4:
        st.subheader("💡 Finding Your Way Back")
        st.error(
            "It sounds like you're having a tough moment. That's okay. "
            "Remember that feelings are temporary. Here are some quick steps you might find helpful right now:"
        )
        
        # Use columns for a nice, side-by-side button layout
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if st.button("Deep Breathing Exercise 🌬️", help="Jump to a guided breathing session.", use_container_width=True):
                set_page("Mindful Breathing")
                st.experimental_rerun()
        
        with col_r2:
            if st.button("Talk to AI Chat 💬", help="Start a supportive conversation with your AI companion.", use_container_width=True):
                set_page("AI Chat")
                st.experimental_rerun()
        
        st.markdown(f"<br>", unsafe_allow_html=True)
        st.info("You can always come back to this page later.")


# ---------- UTILITY: FIREBASE/SUPABASE/INITIALIZATION ----------

def initialize_session_state():
    """Initializes necessary session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["show_splash"] = True
        st.session_state["page"] = "Home"
        st.session_state["user_id"] = None
        st.session_state["ai_chat_history"] = [{"role": "system", "content": "You are a kind, non-judgemental mental wellness companion. Your primary goal is to listen, offer supportive advice, and guide the user toward using the other tools in this app, like journaling or breathing exercises, when appropriate."}]

def app_splash_screen():
    """Displays a splash screen before the main content."""
    st.markdown(
        """
        <style>
        .splash-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            text-align: center;
        }
        .splash-title {
            font-size: 3em;
            font-weight: bold;
            color: #4CAF50; /* A soothing green */
        }
        .splash-loading {
            margin-top: 20px;
            font-size: 1.2em;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="splash-container">', unsafe_allow_html=True)
        st.markdown('<div class="splash-title">ZenFlow Wellness 🌱</div>', unsafe_allow_html=True)
        st.markdown('<p class="splash-loading">Loading your journey...</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Automatically move past splash screen after a short delay
    if st.session_state.get("show_splash"):
        time.sleep(2)
        st.session_state["show_splash"] = False
        st.experimental_rerun()

def unauthenticated_home():
    """Displays the login/guest access page."""
    st.markdown(
        """
        <style>
        .center-content {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 80vh;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container():
        st.markdown('<div class="center-content">', unsafe_allow_html=True)
        st.title("Welcome to ZenFlow Wellness")
        st.subheader("Your digital companion for mental well-being.")
        
        st.markdown("---")
        
        col_login, col_guest = st.columns(2)
        
        with col_login:
            st.button("Login (Mock)", type="primary", use_container_width=True, on_click=lambda: login_mock(True))
        with col_guest:
            st.button("Continue as Guest", use_container_width=True, on_click=lambda: login_mock(False))
        
        st.markdown('</div>', unsafe_allow_html=True)

def login_mock(is_user):
    """Mocks the login process."""
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:8]}" if is_user else "guest"
    st.session_state["page"] = "Home"
    st.experimental_rerun()


def sidebar_navigation():
    """Creates the main app sidebar navigation (Restored to 11 pages)."""
    st.sidebar.title("ZenFlow 🌿")
    st.sidebar.markdown(f"**Logged In:** `{st.session_state['user_id']}`")
    st.sidebar.markdown("---")

    # RESTORED FULL PAGE LIST
    navigation_options = [
        "Home",
        "Mindful Journaling",
        "Mood Tracker",
        "Wellness Check-in",        # Restored
        "CBT Thought Record",
        "Mindful Breathing",
        "AI Chat",
        "Wellness Ecosystem",       # Restored
        "Journal Analysis",         # Restored
        "Report & Summary",
        "IoT Dashboard (ECE)",
    ]

    for page in navigation_options:
        is_selected = st.session_state["page"] == page
        if st.sidebar.button(
            page, 
            key=f"nav_{page}", 
            type="primary" if is_selected else "secondary", 
            use_container_width=True,
        ):
            set_page(page)
            st.experimental_rerun()
            
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.clear()
        initialize_session_state()
        st.experimental_rerun()

# -------------------- MAIN APP EXECUTION --------------------
# Set Streamlit Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="ZenFlow Wellness",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetics and better transitions
st.markdown(
    """
    <style>
    /* Main Content Styling */
    .stApp {
        background-color: #f7f9fc; /* Light background */
    }
    
    /* Sidebar Styling */
    .css-1lcbmhc, .css-1lcbmhc-hover {
        background-color: #ffffff; /* White sidebar */
    }
    .css-1lcbmhc .stButton > button {
        transition: all 0.2s ease-in-out; /* Smooth button transitions */
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        font-weight: 500;
    }
    .css-1lcbmhc .stButton > button:hover {
        background-color: #e6ffe6; /* Light green on hover */
        border-color: #4CAF50;
    }
    .css-1lcbmhc .stButton > button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
    }

    /* Input Fields (Text Areas, Sliders) */
    .stTextArea > div > div > textarea, .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ccc;
        transition: border-color 0.2s;
    }
    .stTextArea > div > div > textarea:focus, .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 0.1rem rgba(76, 175, 80, 0.25);
    }
    
    /* Metric Cards (For the IoT dashboard) */
    .stMetric > div:first-child {
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background-color: #ffffff;
        transition: transform 0.2s;
    }
    .stMetric > div:first-child:hover {
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initialize Session State
initialize_session_state()

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
        
        # Display the selected page content
        if current_page == "Home":
            homepage_panel()
        elif current_page == "Mindful Journaling":
            mindful_journaling_page()
        elif current_page == "Mood Tracker":
            mood_tracker_page()
        elif current_page == "Wellness Check-in": # Restored
            wellness_checkin_page() 
        elif current_page == "CBT Thought Record":
            cbt_thought_record_page()
        elif current_page == "Mindful Breathing":
            mindful_breathing_page()
        elif current_page == "AI Chat":
            ai_chat_page() 
        elif current_page == "Wellness Ecosystem": # Restored
            wellness_ecosystem_page()
        elif current_page == "Journal Analysis": # Restored
            journal_analysis_page()
        elif current_page == "Report & Summary": 
            report_summary_page()
        elif current_page == "IoT Dashboard (ECE)": 
            iot_dashboard_page()
        else:
            st.warning("Page not found or not yet implemented.")
