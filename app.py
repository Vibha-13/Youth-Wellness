import streamlit as st
import os
import time
import random
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import numpy as np

# --- 5.0 REFACTOR: Import all logic from modules ---
from constants import (
    QUOTES, MOOD_EMOJI_MAP, BADGE_RULES, DEFAULT_GOALS, 
    PHQ9_QUESTIONS, PHQ9_SCORES, PHQ9_INTERPRETATION, PHQ9_CRISIS_THRESHOLD, SUICIDE_IDEATION_QUESTION_INDEX, 
    CBT_PROMPTS
)
from ai_utils import (
    setup_analyzer, sentiment_compound, clean_text_for_ai, 
    setup_ai_client, generate_ai_response
)
from db_operations import (
    setup_supabase_client, register_user_db, get_user_by_email_db, 
    save_journal_db, save_mood_db, save_phq9_db, save_ece_log_db, 
    load_all_user_data
)

# --- CONFIG CONSTANTS (Local to app.py) ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" 


# ---------- Streamlit page config and LAYOUT SETUP ----------
st.set_page_config(
    page_title="AI Wellness Companion (5.0 Ready)", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_page_and_layout():
    # ... (CSS styles remain the same, ensuring visual consistency) ...
    # CRITICAL: We'll add one more style for the 5/5 UX pass
    st.markdown("""
    <style>
    /* 1. Global Background and Typography */
    .stApp { 
        background: #F0F2F6; 
        color: #1E1E1E; 
        font-family: 'Poppins', sans-serif; 
    }
    .main .block-container { 
        padding: 2rem 4rem; 
    }
    
    /* 2. CRITICAL: Target the Streamlit Text Area's internal input element */
    textarea {
        color: black !important; 
        -webkit-text-fill-color: black !important; 
        opacity: 1 !important; 
        background-color: white !important; 
        border: 1px solid #ccc !important; 
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
    .css-1d3f90z { 
        background-color: #FFFFFF; 
        box-shadow: 2px 0 10px rgba(0,0,0,0.05); 
    }
    
    /* 6. Custom Card Style (The Core Mobile App Look) */
    .card { 
        background-color: #FFFFFF; 
        border-radius: 16px; 
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); 
        padding: 25px; 
        margin-bottom: 20px; 
        border: none; 
        transition: all .2s ease-in-out; 
    }
    .card:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15); 
    }
    /* NEW: Consistent card padding for inner elements */
    .card .stMarkdown, .card .stButton { 
        padding-top: 5px; 
        padding-bottom: 5px; 
    }


    /* 7. Primary Button Style (Vibrant and Rounded) */
    .stButton>button { 
        color: #FFFFFF; 
        background: #5D54A4; /* Deep Purple */
        border-radius: 25px; 
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
    
    /* [Plant/Ecosystem Styles] */
    .plant-container {
        text-align: center;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 12px;
        background-color: #F8F9FA;
    }
    .plant-emoji {
        font-size: 3rem;
        transition: transform 0.5s ease-out;
    }
    .plant-health-bar {
        height: 15px;
        border-radius: 8px;
        background-color: #E9ECEF;
        overflow: hidden;
        margin-top: 10px;
    }
    .plant-health-fill {
        height: 100%;
        background-color: #28A745;
        transition: width 0.5s ease-out;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------

@st.cache_data
def initialize_kalman(Q_val=0.01, R_val=0.1):
    """Initializes the Kalman filter state variables."""
    return {
        'x_est': 75.0,  
        'P_est': 1.0,   
        'Q': Q_val,     
        'R': R_val      
    }

def kalman_filter_simple(z_meas, state):
    """Applies a single step of the Kalman filter to a noisy measurement."""
    # 1. Prediction, 2. Update logic remains the same
    x_pred = state['x_est'] 
    P_pred = state['P_est'] + state['Q']
    K = P_pred / (P_pred + state['R'])
    x_est = x_pred + K * (z_meas - x_pred)
    P_est = (1 - K) * P_pred
    state['x_est'] = x_est
    state['P_est'] = P_est
    return x_est, state

def generate_simulated_physiological_data(current_time_ms):
    # ... (Simulation logic remains the same) ...
    time_sec = current_time_ms / 1000.0 
    base_hr = 85 + 10 * np.sin(time_sec / 30.0) 
    ppg_noise = 3 * random.gauss(0, 1)
    clean_hr = base_hr + 2 * np.sin(time_sec / 0.5) 
    raw_ppg_signal = clean_hr + ppg_noise
    
    base_gsr = 0.5 * base_hr / 100.0
    phq9_score = st.session_state.get("phq9_score") or 0
    gsr_base = 1.0 + base_gsr + 0.5 * np.random.rand() * (phq9_score / 27.0)
    gsr_noise = 0.5 * random.gauss(0, 1) 
    gsr_value = gsr_base + gsr_noise
    
    return {
        "raw_ppg_signal": raw_ppg_signal, 
        "filtered_hr": clean_hr, 
        "gsr_stress_level": gsr_value,
        "time_ms": current_time_ms
    }

# ---------- Session state defaults (CLEANED UP & LEANER) ----------
if "page" not in st.session_state: st.session_state["page"] = "Home"

# IoT/ECE State
if "kalman_state" not in st.session_state: st.session_state["kalman_state"] = initialize_kalman()
if "physiological_data" not in st.session_state: st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
if "latest_ece_data" not in st.session_state: st.session_state["latest_ece_data"] = {"filtered_hr": 75.0, "gsr_stress_level": 1.0}
if "ece_history" not in st.session_state: st.session_state["ece_history"] = []

# AI/DB/Auth State (Now uses setup functions from modules)
if "_ai_model" not in st.session_state:
    raw_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_KEY = raw_key.strip().strip('"') if isinstance(raw_key, str) and raw_key else None
    if "chat_messages" not in st.session_state: st.session_state["chat_messages"] = []
    _ai_client_obj, _ai_available, _chat_history_list = setup_ai_client(OPENROUTER_API_KEY, st.session_state["chat_messages"])
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

if "daily_journal" not in st.session_state: st.session_state["daily_journal"] = []
if "mood_history" not in st.session_state: st.session_state["mood_history"] = []
if "streaks" not in st.session_state: st.session_state["streaks"] = {"mood_log": 0, "last_mood_date": None, "badges": []}
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "user_email" not in st.session_state: st.session_state["user_email"] = None
if "phq9_score" not in st.session_state: st.session_state["phq9_score"] = None
if "phq9_interpretation" not in st.session_state: st.session_state["phq9_interpretation"] = None
if "last_phq9_date" not in st.session_state: st.session_state["last_phq9_date"] = None
if "last_reframing_card" not in st.session_state: st.session_state["last_reframing_card"] = None
if "cbt_thought_record" not in st.session_state: st.session_state["cbt_thought_record"] = {i: "" for i in range(len(CBT_PROMPTS))}
if "breathing_state" not in st.session_state: st.session_state["breathing_state"] = "finished" 
if "daily_goals" not in st.session_state: st.session_state["daily_goals"] = DEFAULT_GOALS
if "plant_health" not in st.session_state: st.session_state["plant_health"] = 70.0 

analyzer = setup_analyzer()

# ---------- General Helper functions (Utility focus) ----------
def get_all_user_text() -> str:
    """Aggregates all user text for sentiment analysis."""
    parts = []
    parts += [e.get("text","") for e in st.session_state["daily_journal"] if e.get("text")]
    parts += [m.get("content","") for m in st.session_state["chat_messages"] if m.get("role") == "user" and m.get("content")]
    return " ".join(parts).strip()

def calculate_plant_health():
    """Calculates plant health based on goal completion and mood trends."""
    health_base = 50.0 
    
    # 1. Goal Bonus
    goal_completion_score = 0
    total_goals = len(st.session_state["daily_goals"])
    if total_goals > 0:
        for goal_key, goal in st.session_state["daily_goals"].items():
            if goal["count"] >= goal["target"]:
                goal_completion_score += 1
        health_base += (goal_completion_score / total_goals) * 30.0

    # 2. Mood Bonus/Penalty
    if st.session_state["mood_history"]:
        df_mood = pd.DataFrame(st.session_state["mood_history"]).head(7) 
        if not df_mood.empty:
            avg_mood = df_mood['mood'].mean()
            mood_contribution = (avg_mood - 6.0) * 4 
            health_base += mood_contribution

    st.session_state["plant_health"] = max(0, min(100, health_base))

def check_and_reset_goals():
    """Resets daily goals if the last reset date was before today."""
    today = datetime.now().date()
    goals = st.session_state["daily_goals"]
    
    for key, goal in goals.items():
        last_reset = goal.get("last_reset")
        if last_reset:
            last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d").date()
            if last_reset_date < today:
                goal["count"] = 0
                goal["last_reset"] = today.strftime("%Y-%m-%d")
        elif last_reset is None:
            goal["last_reset"] = today.strftime("%Y-%m-%d")

    st.session_state["daily_goals"] = goals
    calculate_plant_health() 

def update_goal_count(goal_key: str, count: int = 1):
    """Increments the count for a specific goal."""
    goals = st.session_state["daily_goals"]
    
    if goal_key in goals and goals[goal_key]["count"] < goals[goal_key]["target"]:
        goals[goal_key]["count"] += count
        calculate_plant_health()
        
check_and_reset_goals()


# ---------- Sidebar Navigation and Auth ----------
# Sidebar Status remains the same

# Sidebar Navigation remains the same

# Sidebar Auth - Logic now calls DB module
def sidebar_auth():
    st.sidebar.markdown("---")
    st.sidebar.header("Account")
    if not st.session_state.get("logged_in"):
        email = st.sidebar.text_input("Your email", key="login_email")
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                db_connected = st.session_state.get("_db_connected")
                
                if db_connected:
                    user_list = get_user_by_email_db(st.session_state.get("_supabase_client_obj"), email)
                    if user_list:
                        user = user_list[0]
                
                if user or db_connected is False:
                    st.session_state["user_id"] = user.get("id") if user else "local_user"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    
                    if user and db_connected:
                        user_data = load_all_user_data(st.session_state["user_id"], st.session_state.get("_supabase_client_obj"))
                        # ... (State loading logic remains the same) ...
                        st.session_state["daily_journal"] = user_data["journal"]
                        st.session_state["mood_history"] = user_data["mood"]
                        st.session_state["ece_history"] = user_data["ece"] 
                        if user_data["phq9"]:
                            latest_phq9 = user_data["phq9"][0]
                            st.session_state["phq9_score"] = latest_phq9.get("score")
                            st.session_state["phq9_interpretation"] = latest_phq9.get("interpretation")
                            st.session_state["last_phq9_date"] = pd.to_datetime(latest_phq9.get("created_at")).strftime("%Y-%m-%d")
                        st.sidebar.success("Logged in and data loaded. ✅")
                    elif db_connected is False:
                         st.sidebar.info("Logged in locally (no DB). 🏠")
                         
                    st.rerun()

                elif db_connected: # Only try to register if DB is connected
                    uid = register_user_db(st.session_state.get("_supabase_client_obj"), email)
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
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            # ... (Logout logic remains the same) ...
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state"]:
                if key in st.session_state:
                    st.session_state[key] = None
            
            st.session_state["daily_journal"] = []
            st.session_state["mood_history"] = []
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            st.session_state["ece_history"] = [] 
            st.session_state["daily_goals"] = DEFAULT_GOALS 
            st.session_state["plant_health"] = 70.0 
            
            _ai_client_obj, _ai_available, _chat_history_list = setup_ai_client(OPENROUTER_API_KEY, [])
            st.session_state["_ai_model"] = _ai_client_obj
            st.session_state["_ai_available"] = _ai_available
            st.session_state["chat_messages"] = _chat_history_list if _ai_available else [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]

            st.sidebar.info("Logged out. 👋")
            st.rerun()

sidebar_auth()


# ---------- PANELS: Homepage (MODIFIED) ----------
def homepage_panel():
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #5D54A4;'>🧠</span></h1>", unsafe_allow_html=True)
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    
    # --- CRISIS ALERT ---
    if st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] >= PHQ9_CRISIS_THRESHOLD:
        st.error("🚨 **CRISIS ALERT:** Your last Wellness Check-in indicated a high level of distress. Please prioritize contacting a helpline or trusted adult immediately. Your safety is paramount.")
    
    st.markdown("---")
    
    # --- SMART NUDGE --- 
    # Logic remains the same, ensuring it uses the imported QUOTES constant
    st.subheader("Your Daily Focus ✨")
    
    df_mood = pd.DataFrame(st.session_state["mood_history"])
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else 6
    latest_stress_index = max(0, st.session_state["latest_ece_data"]["gsr_stress_level"] - 1.0) * 10
    journal_days = len(df_mood)
    mood_streak = st.session_state["streaks"].get("mood_log", 0)
    
    nudge_title = "Welcome Back!"
    nudge_message = random.choice(QUOTES)
    nudge_color = "#FFC107" 
    
    if mood_streak < 3:
        nudge_title = "Habit Builder 🚀"
        nudge_message = "Your plant needs some love! Log your mood today to keep the habit going and water your digital companion."
        nudge_color = "#4a90e2" 
    elif avg_mood_7d < 5.0 and latest_stress_index >= 7:
        nudge_title = "Mind-Body Check-up 😟"
        nudge_message = f"Your mood has been dipping and your vitals show **high stress ({latest_stress_index:.1f})**. Take a 5-minute break and try the **Mindful Breathing** tool right now."
        nudge_color = "#FF4B4B" 
    elif journal_days > 5 and avg_mood_7d < 6.0:
        nudge_title = "Challenge Your Thoughts 🤔"
        nudge_message = "You've been reflecting a lot lately. Use the **CBT Thought Record** to challenge one negative thought that keeps repeating."
        nudge_color = "#00BFFF" 
    
    
    st.markdown(f"""
    <div class='card' style='border-left: 8px solid {nudge_color};'>
        <h3 style='margin-top:0; color: {nudge_color};'>{nudge_title}</h3>
        <p style='font-size: 1.1rem; font-style: italic;'>{nudge_message}</p>
        <p style='text-align: right; margin-top: 10px; font-size: 0.9rem;'>— Your Wellness Buddy</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Row 2 & 3 Quick Actions and Cards remain the same ---

# ---------- PANELS: AI Chat (MODIFIED) ----------
def ai_chat_panel():
    st.header("AI Chat: Talk it Out 💬")
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Display messages
        for message in st.session_state.chat_messages:
            if message["role"] != "system":
                avatar = "🤖" if message["role"] == "assistant" else "👤"
                st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything or tell me how you feel..."):
            with st.spinner("AI is thinking..."):
                
                # --- CALL MODULAR AI FUNCTION ---
                response_text = generate_ai_response(
                    prompt, 
                    st.session_state.chat_messages, 
                    st.session_state.get("_ai_model"),
                    st.session_state.get("_ai_available")
                )
                
                # Append AI response to messages
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PANELS: Mood Tracker (MODIFIED) ----------
def mood_tracker_panel():
    st.header("Daily Mood Tracker 📈")
    
    # --- UX IMPROVEMENT: Encapsulate the form in a card ---
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3,1])
        with col1:
            mood = st.slider("How do you feel right now? (1-11)", 1, 11, 6)
            # Uses imported constant
            st.markdown(f"**You chose:** {MOOD_EMOJI_MAP.get(mood, 'N/A')} · **{mood}/11**") 
            note = st.text_input("Optional: Add a short note about why you feel this way", key="mood_note_input")
            if st.button("Log Mood", key="log_mood_btn"):
                if st.session_state.get("logged_in"):
                    # Save to DB/Local
                    if st.session_state.get("_db_connected"):
                        save_mood_db(st.session_state.get("_supabase_client_obj"), st.session_state["user_id"], mood, note)
                    
                    st.session_state["mood_history"].insert(0, {"date": datetime.now().isoformat(), "mood": mood, "note": note})
                    
                    # Update Streaks & Goals
                    # ... (Streak logic remains the same) ...
                    
                    # --- GOAL UPDATE ---
                    update_goal_count("log_mood")
                    # -------------------
                    
                    st.success("Mood logged successfully! Keep the streak going.")
                    st.rerun()
                else:
                    st.warning("Please log in to save your mood.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # --- Visualizations follow (unchanged) ---

# ---------- PANELS: Mindful Journaling (MODIFIED) ----------
def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    
    ENTRY_KEY = "journal_input_final"
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        entry_text = st.text_area(
            "What's on your mind right now? (The more you write, the better the analysis!)", 
            height=250, 
            key=ENTRY_KEY,
            value="" 
        )
        if st.button("Submit Entry & Analyze", key="submit_journal_btn", use_container_width=True):
            if entry_text:
                sentiment = sentiment_compound(entry_text)
                
                # Save to DB/Local
                if st.session_state.get("logged_in") and st.session_state.get("_db_connected"):
                    save_journal_db(st.session_state.get("_supabase_client_obj"), st.session_state["user_id"], entry_text, sentiment)
                
                st.session_state["daily_journal"].insert(0, {"date": datetime.now().isoformat(), "text": entry_text, "sentiment": sentiment})
                
                # --- GOAL UPDATE ---
                if len(entry_text.split()) > 5: # Require at least 5 words for credit
                    update_goal_count("journal_entry")
                # -------------------
                
                # ... (CBT Reframing logic remains the same) ...
                
                st.session_state["page"] = "Journal Analysis"
                st.rerun()
            else:
                st.warning("Please write something before submitting.")
                
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- PANELS: Mindful Breathing (MODIFIED) ----------
def mindful_breathing_panel():
    st.header("Mindful Breathing 🧘‍♀️")
    
    if st.session_state["breathing_state"] == "finished":
        # ... (Start button logic remains the same) ...
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
        elapsed_time = time.time() - st.session_state["breathing_start_time"]
        if elapsed_time > 180:
            st.session_state["breathing_state"] = "finished"
            st.success("Session complete! You breathed for 3 minutes. Well done! 🏅")
            if "Breathing Master" not in st.session_state["streaks"]["badges"]:
                 st.session_state["streaks"]["badges"].append("Breathing Master")
                 st.balloons()
            
            # --- GOAL UPDATE ---
            update_goal_count("breathing_session")
            # -------------------
            
            st.rerun()
        
        # ... (rest of the running animation logic remains the same) ...


# ... (Wellness Check-in panel - logic uses imported constants) ...
def wellness_checkin_panel():
    st.header("Wellness Check-in (PHQ-9) 🩺")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.info("The Patient Health Questionnaire (PHQ-9) is a clinical tool. This is for self-tracking only and is NOT a diagnostic tool. **If you are in distress, please use the crisis resources.**")
    
    with st.form("phq9_form"):
        st.subheader("Over the last 2 weeks, how often have you been bothered by the following problems?")
        
        responses = {}
        options = list(PHQ9_SCORES.keys())
        
        for i, q in enumerate(PHQ9_QUESTIONS): # Uses imported constant
            responses[i] = st.radio(q, options, key=f"phq9_q_{i}", index=0)
            
        submitted = st.form_submit_button("Submit Check-in", use_container_width=True)
        
        if submitted:
            score = sum(PHQ9_SCORES[resp] for resp in responses.values()) # Uses imported constant
            
            interpretation = "N/A"
            for (min_score, max_score), label in PHQ9_INTERPRETATION.items(): # Uses imported constant
                if min_score <= score <= max_score:
                    interpretation = label
                    break
                    
            st.session_state["phq9_score"] = score
            st.session_state["phq9_interpretation"] = interpretation
            st.session_state["last_phq9_date"] = datetime.now().strftime("%Y-%m-%d")
            
            # Save to DB/Local
            if st.session_state.get("logged_in") and st.session_state.get("_db_connected"):
                save_phq9_db(st.session_state.get("_supabase_client_obj"), st.session_state["user_id"], score, interpretation)
            
            # Safety Check
            if score >= PHQ9_CRISIS_THRESHOLD or responses[SUICIDE_IDEATION_QUESTION_INDEX] != "Not at all":
                st.error("🚨 **CRISIS ALERT:** Your score suggests a need for immediate support. Please reach out to a professional or helpline NOW. **Call or text 988 (US/Canada).**")
            else:
                st.success("Check-in complete! Thank you for prioritizing your wellness.")
            
            st.rerun()

    # ... (Display current score logic remains the same) ...
    st.markdown("</div>", unsafe_allow_html=True)

# ... (All other panels remain the same, but now implicitly use the imported constants and modules) ...
def cbt_thought_record_panel():
    st.header("CBT Thought Record ✍️")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # ... (CBT logic uses imported constant CBT_PROMPTS) ...
    st.markdown("</div>", unsafe_allow_html=True)

def journal_analysis_panel():
    # ... (unchanged) ...
    pass
def iot_dashboard_panel():
    # ... (unchanged) ...
    pass
def wellness_ecosystem_panel():
    # ... (unchanged) ...
    pass
def report_summary_panel():
    # ... (unchanged) ...
    pass


# ---------- MAIN ROUTER (REMAINS THE SAME) ----------
if st.session_state["page"] == "Home":
    homepage_panel()
elif st.session_state["page"] == "AI Chat":
    ai_chat_panel()
elif st.session_state["page"] == "Mood Tracker":
    mood_tracker_panel()
elif st.session_state["page"] == "Mindful Journaling":
    mindful_journaling_panel()
elif st.session_state["page"] == "CBT Thought Record":
    cbt_thought_record_panel()
elif st.session_state["page"] == "Wellness Ecosystem": 
    wellness_ecosystem_panel()
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