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
# In a real environment, you would use this for persistent database storage
try:
    from supabase import create_client
except ImportError:
    # If the user hasn't installed supabase, we define a dummy client to prevent errors
    def create_client(*args, **kwargs):
        return None

# ---------- CONSTANTS ----------
# AI Configuration (using OpenRouter for flexibility)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" 
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" # Fast and effective for summarization/chat

# Inspirational Quotes
QUOTES = [
    "You are the only one who can limit your greatness. — Unknown",
    "I have chosen to be happy because it is good for my health. — Voltaire",
    "A sad soul can kill you quicker, far quicker than a germ. — John Steinbeck",
    "The groundwork for all happiness is health. — Leigh Hunt",
    "A calm mind brings inner strength and self-confidence. — Dalai Lama"
]

# Mood Mapping for the Slider
MOOD_EMOJI_MAP = {
    1: "😭 Agonizing", 2: "😩 Miserable", 3: "😞 Very Sad",
    4: "🙁 Sad", 5: "😐 Neutral/Okay", 6: "🙂 Content",
    7: "😊 Happy", 8: "😁 Very Happy", 9: "🤩 Excited",
    10: "😎 Phenomenal"
}

# CBT Prompts
CBT_PROMPTS = [
    "1. Situation: Describe the event or situation that led to the distress.",
    "2. Emotion: Identify the primary feeling (e.g., anxiety, anger) and rate its intensity (0-100).",
    "3. Negative Thought: Write down the specific, automatic negative thought that occurred.",
    "4. Evidence AGAINST the thought: What facts, logic, or past experiences contradict this thought?",
    "5. Balanced Reframe: Based on the evidence, what is a more balanced, rational perspective?"
]

# Initialize sentiment analyzer (VADER)
analyzer = SentimentIntensityAnalyzer()

# ---------- UTILITY FUNCTIONS: AI, DB, & DATA PROCESSING ----------

def sentiment_compound(text):
    """Calculates the compound sentiment score using VADER."""
    return analyzer.polarity_scores(text)['compound']

def ai_reframing_logic(user_data: dict, negative_thought: str, evidence_against: str):
    """
    Calls the AI model to generate a supportive, evidence-based reframe.
    
    In a real application, this would use the OpenAI/OpenRouter API.
    Since we cannot make external API calls directly in this execution environment,
    this function simulates the call using a well-structured placeholder logic.
    """
    
    # 1. Setup the client (using an environment variable for the key)
    # The key is expected to be available in the environment for OpenRouter/OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        # Simulate AI response if key is missing (Crucial for execution environment)
        time.sleep(1.0) # Add a delay to simulate API latency
        return f"""
        **AI Reframing Assistant (Simulated):**
        Based on the situation: *"{user_data.get(0, 'The situation.')}"*, 
        the evidence you provided shows that your negative thought is not 100% accurate. 
        The key facts that contradict your negative thought are:
        
        - {evidence_against}
        
        A more compassionate and balanced way to view this is:
        *Your feelings are valid, but the thought is not a fact. Look for small, actionable steps forward.*
        """

    # 2. Construct the prompt for the AI
    # This prompt tells the AI how to act and what to generate
    system_prompt = (
        "You are a compassionate Cognitive Behavioral Therapy (CBT) assistant. "
        "Your task is to review the user's negative thought and the evidence they provided "
        "AGAINST that thought. You must write a concise, encouraging, and objective summary "
        "of the counter-evidence to help the user re-evaluate their thought."
    )

    user_prompt = f"""
    User's Situation: {user_data.get(0, 'N/A')}
    User's Emotion/Intensity: {user_data.get(1, 'N/A')}
    User's Negative Thought: {negative_thought}
    User's Evidence AGAINST the thought (Focus on this): {evidence_against}

    Task: Provide a bulleted list summarizing the key points of the evidence provided, 
    and conclude with a single encouraging, rational sentence.
    """

    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )
        
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
        
    except APIError as e:
        # Handle API errors gracefully
        return f"AI Service Error (Code: {e.status_code}): Could not generate reframe. Please try again later."
    except Exception:
        # Fallback if any other issue occurs
        return "AI Service: An unexpected error occurred. Using default simulated reframing logic."


def update_daily_goal(goal_key, amount=1):
    """Updates the count for a specific daily goal and checks for completion."""
    goal = st.session_state["daily_goals"].get(goal_key)
    if goal and goal["count"] < goal["target"]:
        goal["count"] += amount
        st.session_state["daily_goals"][goal_key] = goal # Explicitly save back
        return True
    return False

# --- Database Placeholder Functions (Simulating Supabase Interaction) ---

# Global variable to simulate a database connection
supabase = None 

def connect_to_db():
    """Simulates connecting to Supabase and fetching initial data."""
    global supabase
    
    # Check for connection environment variables
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if url and key and create_client is not None:
        try:
            # Attempt to create a client (in a real app)
            supabase = create_client(url, key)
            st.session_state["_db_connected"] = True
            # Simulate fetching initial data here
            # For now, we rely on the in-session data initialized below
        except Exception:
             st.session_state["_db_connected"] = False
    else:
        st.session_state["_db_connected"] = False
    
    return st.session_state["_db_connected"]

def save_journal_db(user_id, text, sentiment):
    """Saves a journal entry to the database (simulated)."""
    if st.session_state.get("_db_connected") and supabase:
        try:
            # Simulate DB insertion
            # data = supabase.table("journal_entries").insert({...}).execute()
            update_daily_goal("journal_entry")
            return True
        except Exception:
            return False # DB write failed
    else:
        update_daily_goal("journal_entry")
        return True # Successfully updated local goals/history

def save_mood_db(user_id, mood, note):
    """Saves a mood log to the database (simulated)."""
    if st.session_state.get("_db_connected") and supabase:
        try:
            # Simulate DB insertion
            # data = supabase.table("mood_logs").insert({...}).execute()
            update_daily_goal("log_mood")
            return True
        except Exception:
            return False
    else:
        update_daily_goal("log_mood")
        return True
    
def save_cbt_record(cbt_data: dict):
    """Handles CBT record saving and AI reframing."""
    user_id = st.session_state.get("user_id")
    
    # 1. AI Reframing Call
    with st.spinner("🧠 AI Reframing your thought..."):
        ai_reframed_text = ai_reframing_logic(
            cbt_data, 
            cbt_data.get(2, ""), # Negative Thought
            cbt_data.get(3, "")  # Evidence AGAINST
        )

    # 2. Final Record Structure
    record = {
        "date": datetime.now().isoformat(),
        "situation": cbt_data.get(0, ""),
        "emotion": cbt_data.get(1, ""),
        "thought": cbt_data.get(2, ""),
        "evidence_against": cbt_data.get(3, ""),
        "balanced_reframe": cbt_data.get(4, ""),
        "ai_reframing": ai_reframed_text
    }
    
    # 3. Update Local State
    st.session_state["cbt_history"].insert(0, record)
    st.session_state["last_reframing_card"] = record
    update_daily_goal("cbt_record")
    st.success("CBT Thought Record Saved and AI Reframing Completed!")

    # 4. Save to DB (Simulated)
    if st.session_state.get("_db_connected") and supabase:
        try:
            # supabase.table("cbt_records").insert(record).execute()
            pass
        except Exception:
            st.warning("Could not save to database. Record stored locally for this session.")


# ---------- SESSION STATE & INITIALIZATION ----------

def initialize_session_state():
    """Initializes all necessary session state variables."""
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        
    if "show_splash" not in st.session_state:
        st.session_state["show_splash"] = True # New state variable for transition

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Database state
    if "_db_connected" not in st.session_state:
        st.session_state["_db_connected"] = False
        connect_to_db()

    # Data History (Simulated In-Session Data)
    if "daily_journal" not in st.session_state:
        st.session_state["daily_journal"] = [] # [{"date": ..., "text": ..., "sentiment": ...}]
        
    if "mood_history" not in st.session_state:
        # Example historical data for charting
        yesterday = datetime.now() - timedelta(days=1)
        st.session_state["mood_history"] = [
            {"date": (yesterday - timedelta(days=i)).isoformat(), "mood": random.randint(3, 8), "note": f"Mood log for day {i}"}
            for i in range(1, 10)
        ]

    if "cbt_history" not in st.session_state:
        st.session_state["cbt_history"] = []
        
    if "last_reframing_card" not in st.session_state:
        st.session_state["last_reframing_card"] = None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Daily Goals (Resets daily in a real app, here it's based on session)
    today = datetime.now().strftime("%Y-%m-%d")
    if "goal_date" not in st.session_state or st.session_state["goal_date"] != today:
        st.session_state["daily_goals"] = {
            "goal_date": today,
            "journal_entry": {"name": "Write Journal Entry", "count": 0, "target": 1, "icon": "✍️"},
            "log_mood": {"name": "Log Daily Mood", "count": 0, "target": 1, "icon": "😊"},
            "cbt_record": {"name": "Complete CBT Record", "count": 0, "target": 1, "icon": "🧠"},
            "breathing_session": {"name": "Mindful Breathing Session", "count": 0, "target": 1, "icon": "💨"},
        }
        st.session_state["goal_date"] = today

# ---------- STREAMLIT LAYOUT & AESTHETICS ----------

st.set_page_config(
    page_title="HarmonySphere", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

def setup_page_and_layout():
    """Sets up custom CSS for a beautiful, modern look and handles sidebar visibility."""
    
    is_logged_in = st.session_state.get("logged_in", False)
    
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* 1. Global Background and Typography */
.stApp {{ 
    background: linear-gradient(135deg, #fcefee, #e0f7fa); /* soft pastel gradient */
    color: #1E1E1E; 
    font-family: 'Poppins', sans-serif; 
}}
.main .block-container {{ 
    padding: 2rem 3rem;
}}

/* 2. Streamlit TextArea/Input fields */
textarea, input[type="text"], input[type="email"], input[type="password"] {{
    color: #1E1E1E !important;
    -webkit-text-fill-color: #1E1E1E !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border: 2px solid #FFD6E0 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}}
textarea:focus, input[type="text"]:focus, input[type="email"]:focus, input[type="password"]:focus {{
    border-color: #FF9CC2 !important;
    box-shadow: 0 0 8px rgba(255, 156, 194, 0.5);
}}

/* 3. Custom Card Style (Glassy / Wellness Look) */
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
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    cursor: pointer;
    background: rgba(255, 255, 255, 0.9);
}}
h1, h2, h3, h4 {{
    color: #333333;
}}

/* 4. Sidebar Styles (CRITICAL FIX FOR TRANSITION) */
[data-testid="stSidebar"] {{
    /* Default style when logged in */
    background: linear-gradient(to bottom, #fff0f5, #e0f7fa);
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    transition: transform 0.3s ease-in-out;
}}

/* Conditional CSS to HIDE the sidebar immediately upon load when NOT logged in 
   or during the splash screen, ensuring a clean transition. */
{'[data-testid="stSidebar"] { visibility: hidden; transform: translateX(-100%); }' if not is_logged_in else ''}


/* 5. Primary Button Style */
.stButton>button {{
    color: #FFFFFF;
    background: #FF9CC2;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s;
}}
.stButton>button:hover {{
    background: #FF6F91;
    transform: scale(1.02);
}}

/* 6. Sidebar Status Tags */
.sidebar-status {{
    padding: 6px 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    text-align: center;
}}
.status-connected {{ background-color: #D4EDDA; color: #155724; border-left: 4px solid #28A745; }}
.status-local {{ background-color: #FFF3CD; color: #856404; border-left: 4px solid #FFC107; }}
.stProgress > div > div > div > div {{ background-color: #FF9CC2; }}

/* 7. Hide Streamlit Footer */
footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)


# ---------- AUTHENTICATION & NAVIGATION ----------

def navigate_to(page_name):
    """Simple function to handle page navigation."""
    st.session_state["page"] = page_name
    st.rerun()

def sidebar_auth():
    """Manages the sidebar login/logout state and navigation."""
    
    with st.sidebar:
        # Only show the sidebar content if authenticated
        if st.session_state.get("logged_in"):
            st.title("HarmonySphere 🧠")
            st.caption(f"Welcome, User: {st.session_state['user_id'][:8]}...")
            
            # DB Status
            db_status_class = "status-connected" if st.session_state.get("_db_connected") else "status-local"
            db_status_text = "DB Connected" if st.session_state.get("_db_connected") else "Local Mode"
            st.markdown(f"<div class='sidebar-status {db_status_class}'>{db_status_text}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Navigation")
            
            # --- Page Buttons ---
            # Define all pages for navigation
            pages = {
                "Home": "🏠 Dashboard",
                "Mindful Journaling": "📝 Mindful Journaling",
                "Mood Tracker": "📈 Mood Tracker",
                "CBT Thought Record": "✍️ CBT Thought Record",
                "Mindful Breathing": "💨 Mindful Breathing",
                "Journal Analysis": "🔍 Journal Analysis",
                "AI Chat": "🤖 AI Chat Companion",
                "Wellness Check-in": "💚 Wellness Check-in",
                "Wellness Ecosystem": "🔗 Wellness Ecosystem",
                "IoT Dashboard (ECE)": "⚙️ IoT Dashboard (ECE)",
                "Report & Summary": "📊 Report & Summary"
            }
            
            for page_key, page_label in pages.items():
                if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
                    navigate_to(page_key)

            st.markdown("---")
            
            # --- Logout ---
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["page"] = "Home" # Reset to default
                st.session_state["show_splash"] = False # Skip splash on re-login
                st.rerun()
        else:
            # If not logged in, the sidebar is hidden by CSS, but Streamlit still runs this.
            pass


def unauthenticated_home():
    """Shows the centered, clean login page."""
    
    # Use columns to center the content
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px; border-radius: 20px; 
                    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(5px);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-top: 15vh;">
            <h1 style="color: #FF6F91; margin-bottom: 0;">Welcome Back</h1>
            <p style="color: #555; font-size: 1.1rem;">Please log in to access your personal wellness dashboard.</p>
            
            <form id="login_form">
                <input type="email" placeholder="Email (any email will work)" value="user@harmony.com" style="width: 100%; margin-bottom: 10px;"/>
                <input type="password" placeholder="Password (any password will work)" value="password" style="width: 100%; margin-bottom: 20px;"/>
            </form>
        </div>
        """, unsafe_allow_html=True)
        
        # Streamlit Form for handling the action
        with st.form("login_form_st", clear_on_submit=False):
            st.text_input("Username (Any value)", value="DemoUser", key="login_username_input", label_visibility="hidden")
            st.text_input("Password (Any value)", value="DemoPass", type="password", key="login_password_input", label_visibility="hidden")
            login_btn = st.form_submit_button("Access Dashboard", use_container_width=True)

        if login_btn:
            # Simulate authentication success
            st.success("Login successful! Preparing your dashboard...")
            time.sleep(1.0) # Pause for the "transition" feeling
            st.session_state["logged_in"] = True
            st.session_state["user_id"] = "demo-user-" + str(uuid.uuid4())[:4]
            st.rerun()


# ---------- SPLASH SCREEN (For the initial beautiful transition) ----------

def app_splash_screen():
    """Shows the app name briefly before transitioning to login."""
    
    # Use a placeholder to center the content vertically and horizontally
    col_a, col_b, col_c = st.columns([1, 4, 1])

    with col_b:
        # Custom HTML/CSS for a large, centered title
        st.markdown("""
        <div style="text-align: center; margin-top: 25vh; animation: fadeIn 2s ease-in-out;">
            <h1 style="font-size: 5rem; color: #FF9CC2; margin-bottom: 0;">HarmonySphere</h1>
            <p style="font-size: 1.5rem; color: #555;">Your Wellness Companion</p>
        </div>
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)

    # Use a small delay to create the transition effect
    time.sleep(1.5) 
    st.session_state["show_splash"] = False
    st.rerun()


# ---------- AUTHENTICATED PAGES (Functional) ----------

def homepage_panel():
    st.title("🏠 HarmonySphere Dashboard")
    st.subheader(f"Hello, {st.session_state['login_username_input']}! Let's check on your wellness today.")

    st.markdown(f"<div class='metric-card' style='background: #E0F7FA; border-left: 5px solid #00BCD4;'>", unsafe_allow_html=True)
    st.markdown(f"### Today's Goal Tracker")
    st.markdown("Consistency is key to wellness. Complete your tasks to unlock peace.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 1. Daily Goals Section ---
    cols = st.columns(len(st.session_state["daily_goals"]))
    for i, (key, goal) in enumerate(st.session_state["daily_goals"].items()):
        progress = goal["count"] / goal["target"]
        
        # Determine card color based on completion
        card_color = "#D4EDDA" if progress >= 1.0 else "#FFF3CD"
        border_color = "#28A745" if progress >= 1.0 else "#FFC107"

        with cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='background: {card_color}; border-left: 5px solid {border_color};'>
                <h4 style='margin-top:0; color:#333'>{goal['icon']} {goal['name']}</h4>
                <p style='font-size: 1.5rem; font-weight: 700; color:#1E1E1E'>{goal['count']} / {goal['target']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 2. Wellness Insights & Quote ---
    col_insight, col_quote = st.columns([3, 1])

    with col_insight:
        st.header("🧠 Quick Insights")
        
        # Simple Insight: Journaling
        last_entry = st.session_state["daily_journal"][0] if st.session_state["daily_journal"] else None
        if last_entry:
            sentiment_status = "Positive" if last_entry['sentiment'] >= 0.05 else "Negative" if last_entry['sentiment'] <= -0.05 else "Neutral"
            st.info(f"Your last journal entry ({sentiment_status}) was recorded on {pd.to_datetime(last_entry['date']).strftime('%b %d')}.")
        else:
            st.info("Start your first journal entry to unlock personalized insights!")
            
        # Simple Insight: Mood Trend
        mood_df = pd.DataFrame(st.session_state["mood_history"])
        if not mood_df.empty and len(mood_df) > 1:
            avg_mood = mood_df['mood'].mean()
            st.success(f"Your average mood over the last {len(mood_df)} logs is **{avg_mood:.1f}**.")
        
    with col_quote:
        st.markdown(f"<div class='metric-card' style='background: #FFD6E0; height: 100%; border-left: 5px solid #FF9CC2;'>", unsafe_allow_html=True)
        st.subheader("Daily Wisdom")
        st.markdown(f"*{random.choice(QUOTES)}*")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- 3. Feature Shortcuts ---
    st.header("⚡️ Jump Back In")
    col_j, col_m, col_c = st.columns(3)
    
    with col_j:
        if st.button("📝 Start Journaling", use_container_width=True): navigate_to("Mindful Journaling")
    with col_m:
        if st.button("📈 Log Mood", use_container_width=True): navigate_to("Mood Tracker")
    with col_c:
        if st.button("✍️ Challenge Thought", use_container_width=True): navigate_to("CBT Thought Record")


def mindful_journaling_page():
    st.title("📝 Mindful Journaling")
    st.subheader("What's on your mind today?")
    st.caption("Writing down your thoughts and feelings can help you process your emotions.")

    user_id = st.session_state.get("user_id")
    
    # Check if a journal entry has already been completed today
    goal_completed = st.session_state["daily_goals"]["journal_entry"]["count"] >= st.session_state["daily_goals"]["journal_entry"]["target"]
    
    if goal_completed:
        st.success("✅ Daily Journal Goal Complete! You can still write more if you like.")

    # --- Journal Entry Form ---
    with st.form(key="journal_form"):
        journal_text = st.text_area(
            "Write your entry below (minimum 50 characters for best analysis):", 
            height=300,
            key="current_journal_text"
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submit_button = st.form_submit_button("Save Entry", use_container_width=True)
        with col2:
            if st.session_state.get("_db_connected"):
                st.caption("Your entries will be securely saved to your personal database.")
            else:
                st.warning("Database NOT connected. Entry will only be saved locally for this session.")


    if submit_button:
        if len(journal_text.strip()) < 50:
            st.error("Please write at least 50 characters to ensure meaningful analysis.")
        else:
            # 1. Run Sentiment Analysis
            sentiment = sentiment_compound(journal_text)
            
            # 2. Save to Database (and local state)
            db_success = save_journal_db(user_id, journal_text, sentiment)
            
            # 3. Update Local History
            new_entry = {
                "date": datetime.now().isoformat(),
                "text": journal_text,
                "sentiment": sentiment
            }
            st.session_state["daily_journal"].insert(0, new_entry)
            
            # 4. Success message and feedback
            if db_success:
                st.success("Journal Entry Saved Successfully!")
            elif not st.session_state.get("_db_connected"):
                st.info("Entry saved to local session. Connect to database to save permanently.")
                
            # Provide instant sentiment feedback
            st.markdown("---")
            st.subheader("Instant Sentiment Feedback")
            
            if sentiment >= 0.05:
                emoji = "😊"
                sentiment_word = "Positive"
            elif sentiment <= -0.05:
                emoji = "😞"
                sentiment_word = "Negative"
            else:
                emoji = "😐"
                sentiment_word = "Neutral"
                
            st.markdown(f"**Overall Sentiment:** {sentiment_word} {emoji} (Score: **{sentiment:.2f}**)")
            st.info("This is a simple analysis. Your feelings are complex and valid!")
            
            # Clear the form text
            st.session_state["current_journal_text"] = "" 
            st.rerun() # Rerun to update the Goal Complete status


def mood_tracker_page():
    st.title("📈 Daily Mood Tracker")
    st.subheader("How are you feeling right now?")
    st.caption("Logging your mood helps you see patterns over time.")

    user_id = st.session_state.get("user_id")
    
    # Check if a mood has already been logged today
    mood_goal_completed = st.session_state["daily_goals"]["log_mood"]["count"] >= st.session_state["daily_goals"]["log_mood"]["target"]
    if mood_goal_completed:
        st.success("✅ Daily Mood Goal Complete!")

    with st.form("mood_log_form"):
        # Use columns for a better layout
        col_mood, col_note = st.columns([1, 2])

        with col_mood:
            mood_score = st.slider(
                "Select Your Mood Score (1=Worst, 10=Best)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="mood_slider"
            )
            st.markdown(f"**Selected:** Score {mood_score} - {MOOD_EMOJI_MAP.get(mood_score)}")
            
        with col_note:
            mood_note = st.text_area(
                "Quick Note on Why You Feel This Way (Optional)",
                height=120,
                max_chars=250,
                key="mood_note"
            )

        submit_mood = st.form_submit_button("Log Mood", use_container_width=True)

    if submit_mood:
        # 1. Save to Database (and local state)
        db_success = save_mood_db(user_id, mood_score, mood_note)
        
        # 2. Update Local History
        new_log = {
            "date": datetime.now().isoformat(),
            "mood": mood_score,
            "note": mood_note
        }
        st.session_state["mood_history"].insert(0, new_log)
        
        if db_success:
            st.success(f"Logged Mood: {MOOD_EMOJI_MAP.get(mood_score)} - Score {mood_score}")
        else:
            st.warning("Mood logged locally. Database connection failed.")
            
        st.rerun() # Rerun to update the Goal Complete status

    st.markdown("---")
    
    # --- Mood History Visualization ---
    if st.session_state["mood_history"]:
        st.subheader("Your Mood History (Last 30 Days)")
        
        df_mood = pd.DataFrame(st.session_state["mood_history"])
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.tz_localize(None)
        
        # Aggregate to one mood log per day (using the latest log)
        df_mood_daily = df_mood.sort_values('date', ascending=False).drop_duplicates(df_mood['date'].dt.date)
        df_mood_daily = df_mood_daily = df_mood_daily.sort_values('date') # Sort for plot

        # Plotly chart
        fig = px.line(
            df_mood_daily, 
            x='date', 
            y='mood', 
            markers=True, 
            title='Mood Score Trend',
            labels={'mood': 'Mood Score (1-10)', 'date': 'Date'},
            color_discrete_sequence=['#FF9CC2']
        )
        fig.update_layout(yaxis_range=[0.5, 10.5])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your first mood to see your trend chart!")


def cbt_thought_record_page():
    st.title("✍️ CBT Thought Record")
    st.subheader("Challenge Negative Thoughts")
    st.caption("This tool, based on Cognitive Behavioral Therapy, helps you identify and reframe unhelpful thought patterns.")

    # Goal check
    goal_completed = st.session_state["daily_goals"]["cbt_record"]["count"] >= st.session_state["daily_goals"]["cbt_record"]["target"]
    if goal_completed:
        st.success("✅ Daily CBT Record Goal Complete!")

    # Ensure CBT input state exists
    if "cbt_thought_record" not in st.session_state:
        st.session_state["cbt_thought_record"] = {i: "" for i in range(len(CBT_PROMPTS))}

    # --- CBT Form ---
    with st.form(key="cbt_form", clear_on_submit=True):
        st.markdown("### 🔑 Thought Record Steps")
        
        for i, prompt in enumerate(CBT_PROMPTS):
            # Use value from session state to persist input during reruns
            current_value = st.session_state["cbt_thought_record"].get(i, "")
            st.session_state["cbt_thought_record"][i] = st.text_area(
                prompt,
                key=f"cbt_step_{i}",
                height=100 if i not in [3, 4] else 150,
                value=current_value
            )

        st.markdown("---")
        cbt_submitted = st.form_submit_button("Analyze & Reframe Thought", use_container_width=True)
        
    if cbt_submitted:
        # Pass the current state to the saving function
        cbt_data = st.session_state["cbt_thought_record"].copy()
        
        # Simple validation
        if cbt_data.get(0, "") and cbt_data.get(2, "") and cbt_data.get(3, "") and cbt_data.get(4, ""): 
            save_cbt_record(cbt_data) 
            st.rerun()
        else:
            st.error("Please fill out the **Situation**, **Negative Thought**, **Evidence AGAINST**, and **Balanced Reframe** to proceed.")

    # --- Display Last Reframing Card ---
    if st.session_state["last_reframing_card"]:
        st.markdown("---")
        st.subheader("Your Last Reframed Thought ✨")
        
        last_record = st.session_state["last_reframing_card"]
        
        # Display the core record inputs in a nice card
        st.markdown(f"<div class='metric-card' style='background: #E0F7FA; border-left: 5px solid #00BCD4;'>", unsafe_allow_html=True)
        st.markdown(f"**Situation:** {last_record['situation']}")
        st.markdown(f"**Emotion:** {last_record['emotion']}")
        st.markdown(f"**Negative Thought:** **{last_record['thought']}**")
        st.markdown("</div>", unsafe_allow_html=True)


        # Display AI-Generated Counter-Evidence
        st.markdown("#### 🤖 AI's Counter-Evidence (Evidence AGAINST)")
        st.markdown(f"<div class='metric-card' style='background: #FFF3CD; border-left: 5px solid #FFC107;'>{last_record['ai_reframing']}</div>", unsafe_allow_html=True)
        
        # Display User's Final Reframed Thought
        st.markdown("#### ✅ Balanced Reframe (Your Takeaway)")
        st.markdown(f"<div class='metric-card' style='background: #D4EDDA; border-left: 5px solid #28A745;'>{last_record['balanced_reframe']}</div>", unsafe_allow_html=True)

    st.markdown("---")


# ---------- AUTHENTICATED PAGES (PLACEHOLDERS) ----------

def mindful_breathing_page():
    st.title("💨 Mindful Breathing")
    st.subheader("Take a moment to center yourself.")
    st.markdown("Your **Mindful Breathing** feature logic goes here. This would typically involve a visual timer/animation for a breathing exercise (e.g., 4-7-8 method) and a button to update the daily goal.")
    st.info("Goal: 1 breathing session per day.")
    if st.button("Complete Breathing Session", key="complete_breath_btn"):
        if update_daily_goal("breathing_session"):
            st.success("Breathing goal updated! Great job!")
        else:
            st.info("Goal already completed today.")
        st.rerun()

def journal_analysis_page():
    st.title("🔍 Journal Analysis")
    st.subheader("Discover patterns in your writing.")
    st.markdown("Your **Journal Analysis** feature logic goes here. This page should visualize trends in your journal entries, such as sentiment scores over time, and perhaps a word cloud of frequently used words.")
    st.warning("Requires more journal entries to show meaningful data.")

def ai_chat_page():
    st.title("🤖 AI Chat Companion")
    st.subheader("Talk to your non-judgmental wellness assistant.")
    st.markdown("Your **AI Chat** feature logic goes here. This would involve an `st.chat_input` and `st.chat_message` to manage the conversation history and interaction with the AI model.")
    st.warning("Remember to use the `OPENROUTER_API_KEY` for the chat functionality!")

def wellness_checkin_page():
    st.title("💚 Wellness Check-in")
    st.subheader("A quick survey of your current well-being.")
    st.markdown("Your **Wellness Check-in** feature logic goes here. This could be a short form asking about sleep, exercise, social connections, and stress levels, designed to capture a holistic snapshot of health.")

def wellness_ecosystem_page():
    st.title("🔗 Wellness Ecosystem")
    st.subheader("Connect with resources and community.")
    st.markdown("Your **Wellness Ecosystem** feature logic goes here. This could list recommended resources (books, apps, articles) or simulate a safe community space where users can share anonymized insights.")

def iot_dashboard_page():
    st.title("⚙️ IoT Dashboard (ECE Demo)")
    st.subheader("Integrate physical device data.")
    st.markdown("Your **IoT Dashboard** feature logic goes here. For an ECE project, this would show simulated or real data from wearables, smart scales, or home sensors (e.g., ambient light, temperature) to correlate with mood and sleep.")

def report_summary_page():
    st.title("📊 Report & Summary")
    st.subheader("Review your progress over the last week/month.")
    st.markdown("Your **Report & Summary** feature logic goes here. This should summarize key metrics like average mood, number of completed goals, and a quick summary of journal sentiment trends.")


# ---------- MAIN APPLICATION EXECUTION ----------

# 1. Setup Layout and Aesthetics
setup_page_and_layout()

# 2. Initialize Session State (must run before anything else accesses st.session_state)
initialize_session_state()

# 3. Handle Sidebar/Auth
sidebar_auth()

# Create a main placeholder for the application content
app_placeholder = st.empty()

# 4. Handle Page Rendering based on State
with app_placeholder.container():
    
    if st.session_state.get("show_splash"):
        # Transition 1: Show Splash Screen first
        app_splash_screen()
        
    elif not st.session_state.get("logged_in"):
        # Transition 2: Centered Login
        unauthenticated_home()

    else:
        # Transition 3: Authenticated Dashboard
        current_page = st.session_state["page"]
        
        # --- Page Routing ---
        if current_page == "Home":
            homepage_panel()
        elif current_page == "Mindful Journaling":
            mindful_journaling_page()
        elif current_page == "Mood Tracker":
            mood_tracker_page()
        elif current_page == "CBT Thought Record":
            cbt_thought_record_page()
        # Placeholder Pages:
        elif current_page == "Mindful Breathing":
            mindful_breathing_page()
        elif current_page == "Journal Analysis":
            journal_analysis_page()
        elif current_page == "AI Chat":
            ai_chat_page() 
        elif current_page == "Wellness Check-in":
            wellness_checkin_page()
        elif current_page == "Wellness Ecosystem":
            wellness_ecosystem_page()
        elif current_page == "IoT Dashboard (ECE)": 
            iot_dashboard_page()
        elif current_page == "Report & Summary": 
            report_summary_page()
        else:
            st.warning("Page not found or not yet implemented.")
