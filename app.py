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
OPENROUTER_BASE_URL = "https://https://openrouter.ai/api/v1"
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

# [SAFETY CONSTANT]
PHQ9_CRISIS_THRESHOLD = 20 
SUICIDE_IDEATION_QUESTION_INDEX = 8 

# CBT Prompts
CBT_PROMPTS = [
    "**1. The Situation:** What event or trigger led to the strong negative feeling?",
    "**2. The Emotion:** What emotion did you feel? (e.g., Sad, Angry, Anxious, Worthless, Lonely)",
    "**3. The Thought:** What specific automatic negative thought went through your mind? (This is the most crucial part!)",
    "**4. The Evidence FOR the thought:** What facts support your negative thought?",
    "**5. The Evidence AGAINST the thought:** What facts or alternative perspectives go against your negative thought? (Look for exceptions, logic, or other interpretations)",
    "**6. The Balanced Reframe:** What is a more helpful, realistic, and balanced thought you can have right now?"
]

# [NEW CONSTANTS: Goals/Habits]
DEFAULT_GOALS = {
    "log_mood": {"name": "Log Mood", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None},
    "journal_entry": {"name": "Journal Entry", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None},
    "breathing_session": {"name": "Breathing Session", "target": 1, "count": 0, "frequency": "Daily", "last_reset": None}
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
    
    /* [NEW: Plant/Ecosystem Styles] */
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

# Call the setup function early in the main script flow
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
    gsr_base = 1.0 + base_gsr + 0.5 * np.random.rand() * (phq9_score / 27.0)
    gsr_noise = 0.5 * random.gauss(0, 1) # Add some noise to GSR
    gsr_value = gsr_base + gsr_noise
    
    return {
        "raw_ppg_signal": raw_ppg_signal, 
        "filtered_hr": clean_hr, 
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
        return None, False, history 
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL
        )
        
        system_instruction = """
You are 'The Youth Wellness Buddy,' an AI designed for teenagers. 
Your primary goal is to provide non-judgemental, empathetic, and encouraging support. 
Your personality is warm, slightly informal, and very supportive.
Crucially: Always validate the user's feelings first. Never give medical or diagnostic advice. Focus on suggesting simple, actionable coping strategies like breathing, journaling, or connecting with friends. **If a user mentions severe distress, suicidal ideation, or self-harm, immediately pivot to encouraging them to contact a crisis hotline or a trusted adult, and ONLY offer simple, grounding coping methods (like 5-4-3-2-1 technique) until they confirm safety measures are taken. Your priority is safety.** Keep responses concise and focused on the user's current emotional context.
"""
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
if "latest_ece_data" not in st.session_state:
    st.session_state["latest_ece_data"] = {"filtered_hr": 75.0, "gsr_stress_level": 1.0}
if "ece_history" not in st.session_state:
    st.session_state["ece_history"] = []

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
if "cbt_thought_record" not in st.session_state:
    st.session_state["cbt_thought_record"] = {}
    for i in range(len(CBT_PROMPTS)):
        st.session_state["cbt_thought_record"][i] = ""

# Breathing State
if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop" 

# [NEW STATE: Goals/Habits]
if "daily_goals" not in st.session_state:
    st.session_state["daily_goals"] = DEFAULT_GOALS

# [NEW STATE: Plant Gamification]
if "plant_health" not in st.session_state:
    st.session_state["plant_health"] = 70.0 # Start healthy (0-100)

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
    
    # SAFETY CATCH in AI Chat
    if any(phrase in prompt_lower for phrase in ["hurt myself", "end it all", "suicide", "better off dead", "kill myself"]):
        return (
            "**🛑 STOP. This is an emergency.** Please contact help immediately. Your safety is the most important thing. **Call or text 988 (US/Canada) or a local crisis line NOW.** You can also reach out to a trusted family member or teacher. Hold on, you are not alone. Let's try the 5-4-3-2-1 grounding technique together: Name 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste."
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
            context_messages = [messages_for_api[0]] + messages_for_api[-10:]
            
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=context_messages,
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


# ---------- Supabase helpers (DB functions remain the same) ----------
# NOTE: The authentication logic in sidebar_auth() handles fetching the user profile 
# based on Supabase's internal Auth system which is now fixed. 
# The old `users` table logic for login/register is kept but is secondary 
# to the correct profiles table setup.

def register_user_db(email: str):
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return None
    try:
        # Note: This old logic inserts into a separate 'users' table, not the required 'profiles' table.
        # This function is being kept for compatibility with the sidebar_auth() logic you provided.
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
        # Note: This old logic fetches from the secondary 'users' table.
        res = supabase_client.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

# --- NEW JOURNALING SAVE FUNCTION ---
def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception:
        return False

def save_mood_db(user_id, mood: int, note: str) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("mood_logs").insert({"user_id": user_id, "mood_score": mood, "note": note}).execute()
        return True
    except Exception:
        return False

def save_phq9_db(user_id, score: int, interpretation: str) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("phq9_scores").insert({"user_id": user_id, "score": score, "interpretation": interpretation}).execute()
        return True
    except Exception:
        return False
        
def save_ece_log_db(user_id, filtered_hr: float, gsr_stress: float, mood_score: int) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("ece_logs").insert({
            "user_id": user_id, 
            "filtered_hr": filtered_hr, 
            "gsr_stress": gsr_stress,
            "mood_score": mood_score
        }).execute()
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def load_all_user_data(user_id, supabase_client):
    if not supabase_client:
        return {"journal": [], "mood": [], "phq9": [], "ece": []}
    
    data = {}
    try:
        # Load Journal
        res_j = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in res_j.data or []]
        
        # Load Mood
        res_m = supabase_client.table("mood_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["mood"] = [{"date": e.get("created_at"), "mood": e.get("mood_score"), "note": e.get("note")} for e in res_m.data or []]
        
        # Load PHQ-9 (just the latest one)
        res_p = supabase_client.table("phq9_scores").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        data["phq9"] = res_p.data or []

        # Load ECE History
        res_e = supabase_client.table("ece_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["ece"] = [{"date": e.get("created_at"), "hr": e.get("filtered_hr"), "stress": e.get("gsr_stress"), "mood": e.get("mood_score")} for e in res_e.data or []]

    except Exception:
        # Fallback to empty lists on failure
        return {"journal": [], "mood": [], "phq9": [], "ece": []}
        
    return data

# [NEW HELPER FUNCTION: Plant Health]
def calculate_plant_health():
    """Calculates plant health based on goal completion and mood trends."""
    health_base = 50.0 # Start with a neutral baseline
    
    # 1. Goal Bonus (Up to +30)
    goal_completion_score = 0
    total_goals = len(st.session_state["daily_goals"])
    if total_goals > 0:
        for goal_key, goal in st.session_state["daily_goals"].items():
            if goal["count"] >= goal["target"]:
                goal_completion_score += 1
        
        # Max 30 points for completing all goals
        health_base += (goal_completion_score / total_goals) * 30.0

    # 2. Mood Bonus/Penalty (Up to +/- 20)
    if st.session_state["mood_history"]:
        df_mood = pd.DataFrame(st.session_state["mood_history"]).head(7) # Look at last 7 days
        if not df_mood.empty:
            avg_mood = df_mood['mood'].mean()
            # Scale mood (1-11) to health (-20 to +20)
            mood_contribution = (avg_mood - 6.0) * 4 # Max deviation 5 * 4 = 20
            health_base += mood_contribution

    # Clamp health between 0 and 100
    st.session_state["plant_health"] = max(0, min(100, health_base))

    
# [NEW HELPER FUNCTION: Goal Management/Reset]
def check_and_reset_goals():
    """Resets daily goals if the last reset date was before today."""
    today = datetime.now().date()
    
    goals = st.session_state["daily_goals"]
    
    for key, goal in goals.items():
        last_reset = goal.get("last_reset")
        if last_reset:
            last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d").date()
            if last_reset_date < today:
                # Reset for a new day
                goal["count"] = 0
                goal["last_reset"] = today.strftime("%Y-%m-%d")
            # else: continue as it's the same day

        elif last_reset is None:
            # Initialize reset for first time run
            goal["last_reset"] = today.strftime("%Y-%m-%d")

    st.session_state["daily_goals"] = goals
    calculate_plant_health() # Recalculate health after reset


# Run goal check on every app load
check_and_reset_goals()


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
    "CBT Thought Record": "✍️",
    "Wellness Ecosystem": "🌱", # [NEW PAGE]
    "Journal Analysis": "📊",
    "Mindful Breathing": "🧘‍♀️", 
    "IoT Dashboard (ECE)": "⚙️", 
    "Wellness Check-in": "🩺",
    "Report & Summary": "📄"
}

# The navigation choice
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
        # In a real app, you would use st.secrets for Supabase Auth, but this is kept for compatibility with the user's provided logic
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                db_connected = st.session_state.get("_db_connected")
                
                if db_connected:
                    user_list = get_user_by_email_db(email)
                    if user_list:
                        user = user_list[0]
                
                if user or db_connected is False:
                    # Authentication SUCCESS
                    st.session_state["user_id"] = user.get("id") if user else "local_user"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    
                    if user and db_connected:
                        # Load ALL data
                        user_data = load_all_user_data(st.session_state["user_id"], st.session_state.get("_supabase_client_obj"))
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

                else:
                    # Try to register
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
        # Logout logic
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state"]:
                if key in st.session_state:
                    st.session_state[key] = None
            
            # Reset major states
            st.session_state["daily_journal"] = []
            st.session_state["mood_history"] = []
            st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
            st.session_state["kalman_state"] = initialize_kalman()
            st.session_state["ece_history"] = [] 
            st.session_state["daily_goals"] = DEFAULT_GOALS # Reset goals/plant
            st.session_state["plant_health"] = 70.0 
            
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
    
    # --- CRISIS ALERT ---
    if st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] >= PHQ9_CRISIS_THRESHOLD:
        st.error("🚨 **CRISIS ALERT:** Your last Wellness Check-in indicated a high level of distress. Please prioritize contacting a helpline or trusted adult immediately. Your safety is paramount.")
    
    st.markdown("---")
    
    # --- [NEW: SMART NUDGE] Personalized AI Insight ---
    st.subheader("Your Daily Focus ✨")
    
    # Calculate relevant metrics
    df_mood = pd.DataFrame(st.session_state["mood_history"])
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else None
    
    col_nudge, col_quote = st.columns([3, 1])

    with col_nudge:
        with st.container(border=True):
            if st.session_state["daily_goals"].get("journal_entry", {}).get("count", 0) < 1:
                st.info("💡 **Daily Goal:** Haven't journaled today? Take 5 minutes for a quick 'brain dump' on the **Mindful Journaling** page to clear your mind.")
            elif avg_mood_7d is not None and avg_mood_7d < 6:
                st.warning(f"😔 **Mood Check:** Your 7-day average mood score is **{avg_mood_7d:.1f}/11**. Try the **Mindful Breathing** exercise now, or use the **CBT Thought Record** to challenge any stuck negative thoughts.")
            elif st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] > 10:
                st.warning(f"🩺 **Check-in Follow-up:** Your last check-in score was high. Remember to use the tools! The **AI Chat** is always available to listen.")
            else:
                st.success("🎉 **Great work!** You've been engaging well. Remember to check on your Wellness Ecosystem today!")

    with col_quote:
        st.markdown(f"<div class='card' style='padding: 15px;'>", unsafe_allow_html=True)
        st.caption("Quote of the Day")
        st.markdown(f"*{random.choice(QUOTES)}*")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")

    # --- QUICK INSIGHTS ---
    st.subheader("Quick Wellness Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        mood_icon = MOOD_EMOJI_MAP.get(int(avg_mood_7d), "❓") if avg_mood_7d else "❓"
        st.markdown(f"<div class='card'><h3>Average Mood (7D)</h3><h2>{mood_icon} {avg_mood_7d:.1f}/11</h2><p>Past week's emotional stability.</p></div>", unsafe_allow_html=True)

    with col2:
        phq9_text = st.session_state.get("phq9_interpretation") or "Not Taken"
        phq9_color = "green" if phq9_text in ["Minimal to None", "Mild"] else "orange"
        st.markdown(f"<div class='card' style='border-left: 5px solid {phq9_color};'><h3>Last Wellness Score</h3><h2>{st.session_state.get('phq9_score', 'N/A')}</h2><p>{phq9_text} (from last check-in)</p></div>", unsafe_allow_html=True)

    with col3:
        plant_emoji = "🌳" if st.session_state["plant_health"] > 80 else ("🌱" if st.session_state["plant_health"] > 40 else "🌵")
        st.markdown(f"<div class='card'><h3>Ecosystem Health</h3><h2>{plant_emoji} {int(st.session_state['plant_health'])}%</h2><p>Calculated from goal completion & mood.</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Badges & Achievements")
    st.info("You haven't earned any badges yet! Start logging your mood and thoughts to unlock them.")
# --- END of homepage_panel() ---

# --- JOURNALING HELPER (Placement is important: must be before its use) ---
def save_journal_entry(entry_text, user_id, supabase_client):
    """Saves a new journal entry to the Supabase database and updates state."""
    if not entry_text or not user_id:
        st.error("Journal entry cannot be empty.")
        return

    # 1. Perform VADER Sentiment Analysis
    sentiment = sentiment_compound(entry_text)
    
    try:
        # 2. Save to DB
        if st.session_state.get("_db_connected"):
            # Use the dedicated Supabase function
            save_journal_db(user_id, entry_text, sentiment)
        
        # 3. Update Session State (for immediate display)
        new_entry = {"date": datetime.now().isoformat(), "text": entry_text, "sentiment": sentiment}
        st.session_state["daily_journal"].insert(0, new_entry)
        
        # 4. Update Goal Tracker
        st.session_state["daily_goals"]["journal_entry"]["count"] = 1 # Mark as complete for the day
        calculate_plant_health() # Recalculate health

        st.success(f"Journal entry saved! Sentiment Score: {sentiment:.2f} ({'Positive' if sentiment > 0.05 else ('Negative' if sentiment < -0.05 else 'Neutral')})")
        
    except Exception as e:
        st.error(f"Error saving entry: {e}")


# --- Main Journaling Page (Renamed from overview_page) ---
def mindful_journaling_page():
    """Renders the mindful journaling form and recent entries."""
    st.title("Mindful Journaling & Reflection 📝")
    
    user_id = st.session_state.get("user_id")
    supabase_client = st.session_state.get("_supabase_client_obj")

    st.subheader(f"Hello, {st.session_state.get('user_email', 'User')}! Write your thoughts below.")
    
    with st.form("journal_form", clear_on_submit=True):
        journal_entry = st.text_area(
            "What's on your mind today? (Be honest, this is just for you.)",
            height=250,
            placeholder="I feel anxious because..."
        )
        
        # Optional: Add a simple mood rating with the journal entry
        mood_rating = st.slider("Rate your overall mood right now (Optional):", 1, 11, 6, format=f"{MOOD_EMOJI_MAP.get(6).split(' ')[0]} %d")
        
        submitted = st.form_submit_button("Save Reflection")
        
        if submitted:
            if not journal_entry:
                st.warning("Please write something before saving.")
            else:
                # 1. Save the journal entry
                save_journal_entry(journal_entry, user_id, supabase_client)
                
                # 2. Optional: Also save the mood rating if provided (as a separate mood log)
                if st.session_state.get("_db_connected"):
                    save_mood_db(user_id, mood_rating, f"Mood logged via Journal Page (Score: {sentiment_compound(journal_entry):.2f})")
                
                # Rerun to clear the form and update the list
                st.rerun() 

    st.markdown("---")
    st.subheader("Your Recent Entries")

    if st.session_state["daily_journal"]:
        for entry in st.session_state["daily_journal"][:5]: # Show top 5
            sentiment_text = "Positive" if entry["sentiment"] > 0.05 else ("Negative" if entry["sentiment"] < -0.05 else "Neutral")
            sentiment_color = "green" if entry["sentiment"] > 0.05 else ("red" if entry["sentiment"] < -0.05 else "gray")
            
            st.markdown(f"""
            <div class='card' style='border-left: 5px solid {sentiment_color};'>
                <p style='font-size: 0.85rem; color: #777;'>
                    Logged: {pd.to_datetime(entry['date']).strftime('%Y-%m-%d %H:%M')} | Sentiment: <b>{sentiment_text} ({entry['sentiment']:.2f})</b>
                </p>
                <p style='margin-top: 10px;'>{entry['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No journal entries yet. Write your first one above!")
# --- END of mindful_journaling_page() ---


# --- Placeholder Pages (You will build these next) ---

def ai_chat_page():
    st.title("AI Wellness Buddy 💬")
    st.info("This page will host the AI chat interface for real-time support.")
    # (The chat logic is mostly functional in the helpers, but needs a front-end)

def mood_tracker_page():
    st.title("Mood Tracker 📈")
    st.info("This page will feature a slider/form to log your mood and display a time series chart of your emotional state.")

def cbt_thought_record_page():
    st.title("CBT Thought Record ✍️")
    st.info("This page will guide you step-by-step through challenging automatic negative thoughts using the Cognitive Behavioral Therapy prompts.")


# --- MAIN PAGE ROUTER (The final part of your app.py) ---
if st.session_state.get("logged_in") is False:
    # --- LOGOUT/WELCOME PAGE ---
    st.title("Youth Wellness App")
    st.markdown("---")
    st.info("Please use the sidebar to log in or register to access the dashboard features.")
    st.image("https://images.unsplash.com/photo-1558235889-130a08e01087?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Wellness is a journey.", use_column_width=True) 

else:
    # --- AUTHENTICATED PAGES ---
    if st.session_state["page"] == "Home":
        homepage_panel()
    elif st.session_state["page"] == "Mindful Journaling":
        mindful_journaling_page()
    elif st.session_state["page"] == "AI Chat":
        ai_chat_page()
    elif st.session_state["page"] == "Mood Tracker":
        mood_tracker_page()
    elif st.session_state["page"] == "CBT Thought Record":
        cbt_thought_record_page()
    # Add other elif blocks for all your other pages here as you build them
    else:
        # Default fallback for pages not yet defined
        st.header(f"Page: {st.session_state['page']} (Under Construction)")
        st.warning("Feature coming soon!")