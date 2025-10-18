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
    11: "🌟 Fantastic"
}

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

# [Goals/Habits]
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
    
    # --- FIX 1: Defensive check for None state ---
    if state is None:
        state = initialize_kalman() # Re-initialize the Kalman filter state
        st.session_state["kalman_state"] = state
    # ---------------------------------------------
    
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
        # Check if the required secrets are present
        if not url or not key:
             return None, False
        
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
if "ece_running" not in st.session_state:
    st.session_state["ece_running"] = False # Default to stopped

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
    # Safely load secrets, removing surrounding quotes/whitespace if present
    raw_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    raw_key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

    SUPABASE_URL = raw_url.strip().strip('"') if isinstance(raw_url, str) and raw_key else None
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
if "cbt_history" not in st.session_state: # CBT HISTORY STATE
    st.session_state["cbt_history"] = []

# Breathing State
if "breathing_state" not in st.session_state:
    st.session_state["breathing_state"] = "stop" 

# [Goals/Habits]
if "daily_goals" not in st.session_state:
    st.session_state["daily_goals"] = DEFAULT_GOALS.copy()

# [Plant Gamification]
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

        # Append new user message before sending to API if the context is for the chat page
        is_chat_context = st.session_state["page"] == "AI Chat"
        if is_chat_context and messages_for_api and (messages_for_api[-1]["content"] != prompt_clean or messages_for_api[-1]["role"] != "user"):
            messages_for_api.append({"role": "user", "content": prompt_clean})
        elif not is_chat_context:
            # If not in chat, use a minimal context for the specific task (like CBT reframing)
            # Use the existing system prompt from session state if available, otherwise just use the new user prompt
            system_prompt = st.session_state.chat_messages[0]["content"] if st.session_state.chat_messages and st.session_state.chat_messages[0]["role"] == "system" else "You are a helpful AI assistant."
            messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_clean}]


        try:
            # Use only the last 10 messages for context, plus the system prompt
            context_messages = [messages_for_api[0]] + messages_for_api[-10:] if len(messages_for_api) > 1 else messages_for_api
            
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL_NAME,
                messages=context_messages,
                max_tokens=max_tokens,
                temperature=0.7 
            )
            
            if resp.choices and resp.choices[0].message:
                return resp.choices[0].message.content
            
        except APIError:
            # Fallback on API Error
            pass
        except Exception:
            # General fallback
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

# ---------- Supabase helpers (DB functions remain the same) ----------

def register_user_db(email: str):
    """
    Inserts a new user entry into the 'users' and 'profiles' tables 
    with a generated UUID and current timestamp.
    """
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return None
        
    # 1. Generate a valid UUID for the new user ID
    new_user_id = str(uuid.uuid4())
    
    # 2. Get current timestamp in ISO format for PostgreSQL
    current_time = datetime.now().isoformat() 
    
    try:
        # 3. Insert into 'users' table (CRITICAL: Includes id and created_at)
        res_user = supabase_client.table("users").insert({
            "id": new_user_id,
            "email": email,
            "created_at": current_time 
        }).execute()

        # 4. Also insert into 'profiles' table (CRITICAL: Includes id and created_at)
        supabase_client.table("profiles").insert({
            "id": new_user_id,
            "username": email.split("@")[0],
            "created_at": current_time
        }).execute()
        
        # Check if the user insert was successful
        if getattr(res_user, "data", None):
            return new_user_id
            
    except Exception as e:
        st.error(f"DB Error: {e}") # Uncomment this line temporarily to see the real error if it still fails
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

# --- SAVE FUNCTIONS ---
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
        
# --- CBT SAVE FUNCTION ---
def save_cbt_record(cbt_data: dict):
    """Saves the CBT record and generates the AI's counter-evidence."""
    
    # 1. Pull the crucial entries for AI prompt
    situation = cbt_data.get(0, "")
    emotion = cbt_data.get(1, "")
    negative_thought = cbt_data.get(2, "")
    evidence_for = cbt_data.get(3, "")
    balanced_reframe = cbt_data.get(5, "")
    
    if not situation or not negative_thought:
        st.error("Please fill out the Situation and Negative Thought fields.")
        return False
        
    # 2. Construct the AI prompt for the "Evidence AGAINST" step
    ai_prompt = f"""
    A user is completing a CBT Thought Record. Their situation was: "{situation}". 
    They felt: "{emotion}". Their core automatic negative thought was: "{negative_thought}". 
    The user's evidence FOR this thought is: "{evidence_for}". 
    
    Your task is to act as a supportive CBT therapist. Generate a concise, objective, 
    and non-judgemental list of **3-4 logical counter-arguments and alternative perspectives** (Evidence AGAINST the negative thought). 
    Use bullet points and encouraging language. Start with 'Here are some facts or alternative ways to look at this:'
    """
    
    # 3. Get AI Counter-Evidence
    ai_reframing_text = safe_generate(ai_prompt, max_tokens=400)
    
    # 4. Finalize the record for saving/display (local state)
    record = {
        "id": time.time(),
        "date": datetime.now().isoformat(),
        "situation": situation,
        "emotion": emotion,
        "thought": negative_thought,
        "evidence_for": evidence_for,
        "ai_reframing": ai_reframing_text, # AI-generated part
        "balanced_reframe": balanced_reframe # User-input final step
    }

    # 5. Update local state
    if "cbt_history" not in st.session_state:
        st.session_state["cbt_history"] = []
        
    st.session_state["cbt_history"].insert(0, record)
    st.session_state["cbt_thought_record"] = {i: "" for i in range(len(CBT_PROMPTS))} # Clear form
    
    st.success("Thought Record completed and reframed! Review the AI's counter-evidence below.")
    st.session_state["last_reframing_card"] = record
    return True

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

def calculate_plant_health():
    """Calculates plant health based on goal completion and mood trends."""
    health_base = 50.0 # Start with a neutral baseline
    
    # 1. Goal Bonus (Up to +30)
    goal_completion_score = 0
    
    # Defensive check for daily_goals being set
    goals = st.session_state.get("daily_goals")
    if goals is None:
        goals = DEFAULT_GOALS.copy()
        st.session_state["daily_goals"] = goals

    total_goals = len(goals)
    if total_goals > 0:
        for goal_key, goal in goals.items():
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
    
def check_and_reset_goals():
    """Resets daily goals if the last reset date was before today."""
    today = datetime.now().date()
    
    # FIX 1: Ensure daily_goals is a dictionary before proceeding (Resolves original AttributeError)
    if st.session_state.get("daily_goals") is None:
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()

    goals = st.session_state["daily_goals"]
    
    for key, goal in goals.items():
        last_reset = goal.get("last_reset")
        if last_reset:
            # Safely parse the date, handling potential errors
            try:
                last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d").date()
            except ValueError:
                # If date format is wrong, reset it to today
                last_reset_date = today - timedelta(days=1) 
                
            if last_reset_date < today:
                # Reset for a new day
                goal["count"] = 0
                goal["last_reset"] = today.strftime("%Y-%m-%d")

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
    f"<div class='sidebar-status {ai_status_class}'>AI: <b>{'CONNECTED (OpenRouter)' if st.session_state.get('_ai_available') else 'LOCAL (fallback)'}</b></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<div class='sidebar-status {db_status_class}'>DB: <b>{'CONNECTED' if st.session_state.get('_db_connected') else 'NOT CONNECTED'}</b></div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Sidebar Navigation (FULL LIST)
st.sidebar.header("Navigation")
page_options = {
    "Home": "🏠", 
    "AI Chat": "💬", 
    "Mood Tracker": "📈", 
    "Mindful Journaling": "📝", 
    "CBT Thought Record": "✍️",
    "Wellness Ecosystem": "🌱", 
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
        st.sidebar.caption("Use only your email to log in or register locally.")
        
        email = st.sidebar.text_input("Your email", key="login_email_input_fix").lower().strip()
        submitted = st.sidebar.button("Access Dashboard", key="login_button_fix", use_container_width=True)
            
        if submitted:
            if email and "@" in email:
                # Clear existing session data before logging in
                for key in ["user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state", "daily_journal", "mood_history", "physiological_data", "ece_history", "plant_health", "cbt_history", "last_reframing_card"]:
                    if key in st.session_state:
                        st.session_state[key] = None
                        
                user = None
                db_connected = st.session_state.get("_db_connected")
                
                # --- Login/Lookup Attempt ---
                if db_connected:
                    user_list = get_user_by_email_db(email)
                    if user_list:
                        user = user_list[0]
                
                if user or db_connected is False:
                    # Authentication SUCCESS (either via DB lookup or local fallback)
                    st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    
                    if user and db_connected:
                        # Load ALL data from DB
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
                        st.session_state["daily_goals"] = DEFAULT_GOALS.copy() # Ensure goals are reset/set for local user
                        st.sidebar.info("Logged in locally (no DB). 🏠")
                        
                    st.rerun()

                else:
                    # --- Registration Attempt ---
                    if db_connected:
                        uid = register_user_db(email)
                        if uid:
                            st.session_state["user_id"] = uid
                            st.session_state["user_email"] = email
                            st.session_state["logged_in"] = True
                            st.sidebar.success("Registered & logged in. 🎉")
                            st.rerun()
                        else:
                            st.sidebar.error("Registration failed. Please check DB connection or RLS rules.")
                    else:
                            st.sidebar.error("Could not find user and DB not connected to register.")

            else:
                st.sidebar.warning("Please enter a valid email address.")
    else:
        # Logout logic
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout", key="sidebar_logout_btn"):
            # Reset major state variables
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state", "daily_journal", "mood_history", "physiological_data", "ece_history", "daily_goals", "plant_health", "chat_messages", "cbt_history", "last_reframing_card"]:
                if key in st.session_state:
                    st.session_state[key] = None
                    
            st.session_state["daily_goals"] = DEFAULT_GOALS.copy() # Use copy() for safety
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


# ---------- FEATURE PAGE FUNCTIONS (Built and working) ----------

# --- 1. Homepage ---
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
    df_mood = pd.DataFrame(st.session_state["mood_history"]) if st.session_state.get("mood_history") else pd.DataFrame()
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else None
    
    # --- Check if mood data exists for formatting ---
    if avg_mood_7d is None:
        avg_mood_display = "N/A"
        mood_icon = "❓"
    else:
        avg_mood_display = f"{avg_mood_7d:.1f}"
        mood_icon = MOOD_EMOJI_MAP.get(int(round(avg_mood_7d)), "❓") 
    
    col_nudge, col_quote = st.columns([3, 1])

    with col_nudge:
        with st.container(border=True):
            if st.session_state.get("daily_goals", {}).get("journal_entry", {}).get("count", 0) < 1:
                st.info("💡 **Daily Goal:** Haven't journaled today? Take 5 minutes for a quick 'brain dump' on the **Mindful Journaling** page to clear your mind.")
            elif avg_mood_7d is not None and avg_mood_7d < 6:
                st.warning(f"😔 **Mood Check:** Your 7-day average mood score is **{avg_mood_display}/11**. Try the **Mindful Breathing** exercise now, or use the **CBT Thought Record** to challenge any stuck negative thoughts.")
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
        st.markdown(f"<div class='card'><h3>Average Mood (7D)</h3><h2>{mood_icon} {avg_mood_display}{'/11' if avg_mood_7d is not None else ''}</h2><p>Past week's emotional stability.</p></div>", unsafe_allow_html=True)

    with col2:
        phq9_text = st.session_state.get("phq9_interpretation") or "Not Taken"
        phq9_score_display = st.session_state.get('phq9_score', 'N/A')
        phq9_color = "green" if phq9_text in ["Minimal to None", "Mild"] else ("orange" if phq9_score_display != 'N/A' and phq9_score_display is not None and phq9_score_display >= 10 else "gray")
        
        st.markdown(f"<div class='card' style='border-left: 5px solid {phq9_color};'><h3>Last Wellness Score</h3><h2>{phq9_score_display}</h2><p>{phq9_text} (from last check-in)</p></div>", unsafe_allow_html=True)

    with col3:
        plant_health_int = int(st.session_state.get('plant_health', 70.0))
        plant_emoji = "🌳" if plant_health_int > 80 else ("🌱" if plant_health_int > 40 else "🌵")
        st.markdown(f"<div class='card'><h3>Ecosystem Health</h3><h2>{plant_emoji} {plant_health_int}%</h2><p>Calculated from goal completion & mood.</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Badges & Achievements")
    st.info("Badges are coming soon: Track your streaks and engagement here!")

# --- Mood Tracker Helper ---
def update_mood_streak():
    """Checks the mood history to calculate the current logging streak."""
    
    if not st.session_state.get("mood_history"): # Use .get() and check truthiness
        st.session_state["streaks"]["mood_log"] = 0
        return
        
    df = pd.DataFrame(st.session_state["mood_history"])
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    logged_dates = sorted(df['date_only'].unique(), reverse=True)
    
    if not logged_dates:
        st.session_state["streaks"]["mood_log"] = 0
        return

    today = datetime.now().date()
    streak = 0
    current_date = today

    # Check if a log exists for today
    has_logged_today = logged_dates[0] == today if logged_dates and logged_dates[0] == today else False
    
    if has_logged_today:
        streak += 1
        current_date = current_date - timedelta(days=1)
        logged_dates = logged_dates[1:] 

    # Iterate backwards through logged days to count the streak
    for log_date in logged_dates:
        expected_date = current_date
        
        if log_date == expected_date:
            streak += 1
            current_date = current_date - timedelta(days=1)
        elif log_date < expected_date:
            break
            
    st.session_state["streaks"]["mood_log"] = streak
    
    # Defensive check for daily_goals being set
    if st.session_state.get("daily_goals") is None:
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
        
    if has_logged_today:
        st.session_state["daily_goals"]["log_mood"]["count"] = 1
    
    calculate_plant_health() 

# --- 2. Mindful Journaling ---
def mindful_journaling_page():
    st.title("Mindful Journaling & Reflection 📝")
    
    user_id = st.session_state.get("user_id")
    supabase_client = st.session_state.get("_supabase_client_obj")

    st.subheader(f"Hello, {st.session_state.get('user_email', 'User')}! Write your thoughts below.")
    
    # Journal Helper function defined locally for context
    def save_journal_entry(entry_text, user_id, supabase_client, mood_rating, mood_note):
        if not entry_text or not user_id:
            st.error("Journal entry cannot be empty or user not logged in.")
            return

        sentiment = sentiment_compound(entry_text)
        
        try:
            if st.session_state.get("_db_connected"):
                save_journal_db(user_id, entry_text, sentiment)
            
            new_entry = {"date": datetime.now().isoformat(), "text": entry_text, "sentiment": sentiment}
            
            # Defensive check for daily_journal being set to None
            if st.session_state.get("daily_journal") is None:
                st.session_state["daily_journal"] = []
                
            st.session_state["daily_journal"].insert(0, new_entry)
            
            if mood_rating and user_id:
                if st.session_state.get("_db_connected"):
                    save_mood_db(user_id, mood_rating, f"Mood logged via Journal Page ({mood_note or 'No note'})")
                
                new_mood_entry = {"date": datetime.now().isoformat(), "mood": mood_rating, "note": f"Journal Mood: {mood_note}"}

                # Defensive check for mood_history being set to None
                if st.session_state.get("mood_history") is None:
                    st.session_state["mood_history"] = []
                    
                st.session_state["mood_history"].insert(0, new_mood_entry)
                update_mood_streak()

            # Defensive check for daily_goals being set
            if st.session_state.get("daily_goals") is None:
                st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
                
            st.session_state["daily_goals"]["journal_entry"]["count"] = 1
            calculate_plant_health() 

            st.success(f"Journal entry saved! Sentiment Score: {sentiment:.2f} ({'Positive' if sentiment > 0.05 else ('Negative' if sentiment < -0.05 else 'Neutral')})")
            
        except Exception as e:
            st.error(f"Error saving entry: {e}")

    # We must use a form here for the text_area to be clearable on submit
    with st.form("journal_form", clear_on_submit=True):
        journal_entry = st.text_area(
            "What's on your mind today? (Be honest, this is just for you.)",
            height=250,
            placeholder="I feel anxious because..."
        )
        
        mood_note = st.text_input("Quick summary of your current mood for the optional rating (e.g. 'Feeling great after finishing the code').")
        
        mood_rating = st.slider("Rate your overall mood right now (Optional - Logs to Mood Tracker):", 1, 11, 6, format=f"{MOOD_EMOJI_MAP.get(6).split(' ')[0]} %d", key="journal_mood_slider")
        
        submitted = st.form_submit_button("Save Reflection")
        
        if submitted:
            if not journal_entry:
                st.warning("Please write something before saving.")
            else:
                save_journal_entry(journal_entry, user_id, supabase_client, mood_rating, mood_note)
                st.rerun() 

    st.markdown("---")
    st.subheader("Your Recent Entries")

    if st.session_state.get("daily_journal"):
        for entry in st.session_state["daily_journal"][:5]: 
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


# --- 3. Mood Tracker ---
def mood_tracker_page():
    st.title("Mood Tracker & Analysis 📈")
    st.markdown("Log your current emotional state and review your trends over time.")
    
    user_id = st.session_state.get("user_id")
    supabase_client = st.session_state.get("_supabase_client_obj")

    # --- MOOD LOGGING FORM ---
    st.subheader("How are you feeling right now?")
    with st.container(border=True):
        col_slider, col_mood_display = st.columns([3, 1])
        
        with col_slider:
            mood_score = st.slider(
                "Select your mood score (1=Agonizing, 11=Joyful)",
                1, 11, 6,
                key="current_mood_slider"
            )
        
        with col_mood_display:
            mood_text = MOOD_EMOJI_MAP.get(mood_score, "❓ Unknown")
            st.markdown(f"**Your Choice:**<br/><h2>{mood_text}</h2>", unsafe_allow_html=True)
            
        mood_note = st.text_input(
            "Quick note on why you feel this way (optional)",
            placeholder="E.g., I'm happy because I finished my assignment, or, I'm stressed about the test tomorrow."
        )

        if st.button("Log My Mood", use_container_width=True):
            if not user_id:
                st.error("Please log in to save your mood.")
            else:
                # 1. Save to DB
                if st.session_state.get("_db_connected"):
                    save_mood_db(user_id, mood_score, mood_note)

                # 2. Update Session State (for immediate display)
                new_entry = {"date": datetime.now().isoformat(), "mood": mood_score, "note": mood_note}
                
                # --- FIX 2: Defensive check for None state (prevents AttributeError) ---
                if st.session_state.get("mood_history") is None:
                    st.session_state["mood_history"] = []
                # ---------------------------------------------------------------------

                st.session_state["mood_history"].insert(0, new_entry)
                
                # 3. Update Streak & Goals
                update_mood_streak()

                st.success(f"Mood logged as: {mood_text}!")
                st.rerun() 
                
    st.markdown("---")

    # --- MOOD STREAK AND HISTORY ---
    col_streak, col_badge = st.columns([1, 2])
    
    update_mood_streak() # Ensure streak is calculated on load
    with col_streak:
        current_streak = st.session_state["streaks"]["mood_log"]
        st.markdown(f"<div class='card' style='padding: 15px; text-align: center;'><h3>🔥 Current Streak</h3><h1>{current_streak} days</h1><p>Consecutive days logging your mood.</p></div>", unsafe_allow_html=True)

    with col_badge:
        st.info("Badges are coming soon: Track your streaks and engagement here!")
        
    st.markdown("---")
    
    # --- MOOD TREND CHART ---
    st.subheader("Your Mood Trends Over Time")

    if st.session_state.get("mood_history"):
        df_mood = pd.DataFrame(st.session_state["mood_history"])
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        
        df_mood = df_mood.sort_values(by='date').tail(30).reset_index(drop=True) 

        fig = px.line(
            df_mood,
            x='date',
            y='mood',
            title='Last 30 Mood Logs',
            markers=True,
            height=400
        )
        
        fig.update_layout(
            yaxis=dict(
                tickvals=list(MOOD_EMOJI_MAP.keys()),
                ticktext=[f"{emoji.split(' ')[0]} {score}" for score, emoji in MOOD_EMOJI_MAP.items()],
                range=[0.8, 11.2],
                title="Mood Score"
            ),
            xaxis_title="Date/Time",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Start logging your mood above to see your trend chart!")

# --- 4. AI Chat ---
def ai_chat_page():
    st.title("AI Wellness Buddy 💬")
    st.markdown("I'm here to listen without judgment. How can I support you today?")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input logic
    if prompt := st.chat_input("Say something..."):
        # Append user message to display immediately
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Buddy is thinking..."):
            response = safe_generate(prompt)
            
        # Append assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# --- 5. Wellness Check-in (PHQ-9) ---
def interpret_phq9_score(score: int) -> str:
    """Interprets the total PHQ-9 score based on standard clinical ranges."""
    for score_range, interpretation in PHQ9_INTERPRETATION.items():
        if score_range[0] <= score <= score_range[1]:
            return interpretation
    return "Unknown"

def wellness_checkin_page():
    st.title("Wellness Check-in (PHQ-9) 🩺")
    st.markdown("This quick assessment helps track your mood over the past **two weeks**.")
    st.markdown("---")
    
    last_date = st.session_state.get("last_phq9_date")
    if last_date:
        st.info(f"Your last check-in was on **{last_date}**. Your score was **{st.session_state.get('phq9_score') or 'N/A'}** ({st.session_state.get('phq9_interpretation') or 'N/A'}).")

    st.subheader("Over the last two weeks, how often have you been bothered by the following problems?")
    
    phq9_responses = {}
    
    with st.form("phq9_form"):
        
        for i, question in enumerate(PHQ9_QUESTIONS):
            key = f"q_{i}"
            
            st.markdown(f"**{question}**")
            
            response = st.radio(
                options=list(PHQ9_SCORES.keys()),
                key=key,
                index=0,
                horizontal=True,
                label="Select one:" # Added label
            )
            
            phq9_responses[i] = response
            st.markdown("---")


        submitted = st.form_submit_button("Submit Check-in")

        if submitted:
            total_score = 0
            for i, response in phq9_responses.items():
                total_score += PHQ9_SCORES[response]
                
            interpretation = interpret_phq9_score(total_score)
            user_id = st.session_state.get("user_id")
            
            if st.session_state.get("_db_connected"):
                save_phq9_db(user_id, total_score, interpretation)
                
            st.session_state["phq9_score"] = total_score
            st.session_state["phq9_interpretation"] = interpretation
            st.session_state["last_phq9_date"] = datetime.now().strftime("%Y-%m-%d")
            
            st.subheader(f"✅ Check-in Complete! Your score is **{total_score}** ({interpretation}).")
            
            is_high_risk = (total_score >= PHQ9_CRISIS_THRESHOLD) or (PHQ9_SCORES[phq9_responses[SUICIDE_IDEATION_QUESTION_INDEX]] >= 1)

            if is_high_risk:
                st.error("""
                🚨 **IMPORTANT: High Distress Indication!**
                Your responses indicate significant distress. Please know you are not alone.
                **Action:** Contact a crisis line or a trusted adult immediately.
                **Call or text 988 (US/Canada) or a local emergency number NOW.**
                """, icon="🚨")
            elif total_score >= 10:
                st.warning(f"Your score is in the **{interpretation}** range. This is a good time to use the **CBT Thought Record** or talk to the **AI Chat**.")
            else:
                st.success("Your score is low. Keep up the good work on your self-care!")
                
            st.rerun()

# --- 6. Wellness Ecosystem ---
def wellness_ecosystem_page():
    st.title("Your Wellness Ecosystem 🌱")
    st.markdown("Your virtual plant thrives when you take care of your mental health! Complete your daily goals to help it grow.")
    st.markdown("---")
    
    col_plant, col_goals = st.columns([1, 2])
    
    # 1. Plant Health Status
    with col_plant:
        plant_health = st.session_state.get("plant_health", 70.0)
        plant_health_int = int(plant_health)
        
        # Determine plant emoji based on health
        if plant_health_int > 85:
            plant_emoji = "🌸🌳"
            caption = "Thriving! Keep up the excellent work!"
        elif plant_health_int > 50:
            plant_emoji = "🌱"
            caption = "Healthy. Water your mind with self-care!"
        elif plant_health_int > 20:
            plant_emoji = "🌿"
            caption = "Needs attention. Focus on one small goal."
        else:
            plant_emoji = "🌵"
            caption = "Struggling. Reach out to the AI Buddy or check in with yourself."
        
        # Determine health color for the progress bar
        bar_color = '#28A745' if plant_health > 50 else ('#FFC107' if plant_health > 20 else '#DC3545')

        st.markdown(f"""
        <div class='plant-container'>
            <div class='plant-emoji' style='transform: scale({1 + plant_health/150});'>
                {plant_emoji}
            </div>
            <h3>Health: {plant_health_int}%</h3>
            <p style='font-size: 0.9rem; color: #555;'>{caption}</p>
            <div class='plant-health-bar'>
                <div class='plant-health-fill' style='width: {plant_health_int}%; background-color: {bar_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Daily Goals Tracker
    with col_goals:
        st.subheader("Your Daily Goals")
        
        # Defensive check for daily_goals being set
        if st.session_state.get("daily_goals") is None:
            st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
            
        goals = st.session_state["daily_goals"]
        
        for key, goal in goals.items():
            is_complete = goal["count"] >= goal["target"]
            icon = "✅" if is_complete else "⏳"
            
            st.markdown(f"""
            <div class='card' style='padding: 15px; border-left: 5px solid {'#28A745' if is_complete else '#5D54A4'};'>
                <h4>{icon} {goal['name']}</h4>
                <p>Status: {'Completed!' if is_complete else f"Needs: {goal['count']}/{goal['target']} Time"}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.info("Goals update automatically when you log your mood, journal, or complete a breathing session.")

# --- 7. Mindful Breathing ---
def mindful_breathing_page():
    st.title("Mindful Breathing 🧘‍♀️")
    st.markdown("Follow the animation to slow your breathing and calm your nervous system. (4 seconds in, 6 seconds out).")
    st.markdown("---")

    col_btn, col_spacer = st.columns([1, 2])

    with col_btn:
        if st.session_state["breathing_state"] == "stop":
            if st.button("Start 3-Minute Session", use_container_width=True):
                st.session_state["breathing_state"] = "running"
                st.session_state["breathing_start_time"] = time.time()
                st.session_state["breathing_duration"] = 3 * 60 # 3 minutes
                
                # Defensive check for daily_goals being set
                if st.session_state.get("daily_goals") is None:
                    st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
                    
                st.session_state["daily_goals"]["breathing_session"]["count"] = 1
                calculate_plant_health()
                st.rerun()
        else:
            if st.button("Stop Session", type="secondary", use_container_width=True):
                st.session_state["breathing_state"] = "stop"
                st.rerun()
    
    # Breathing Session Logic
    if st.session_state["breathing_state"] == "running":
        elapsed_time = time.time() - st.session_state["breathing_start_time"]
        remaining_time = st.session_state["breathing_duration"] - elapsed_time

        if remaining_time <= 0:
            st.session_state["breathing_state"] = "stop"
            st.success("🎉 Breathing session complete! You did a great job.")
            st.rerun()
            return
            
        st.info(f"Time Remaining: {int(remaining_time // 60)}:{int(remaining_time % 60):02d}")
        
        # Determine the phase and animation class
        cycle_time = elapsed_time % 10 # 4s in + 6s out = 10s cycle
        
        if 0 <= cycle_time < 4:
            phase = "Inhale"
            animation_class = "breathe-inhale"
        else:
            phase = "Exhale"
            animation_class = "breathe-exhale"
        
        # Display the animated circle and instruction
        st.markdown(f"""
        <div class='breathing-circle {animation_class}'>
            {phase}
        </div>
        """, unsafe_allow_html=True)
        
        # Rerun quickly to update the animation and timer (100ms)
        time.sleep(0.1)
        st.rerun()
        
    elif st.session_state["breathing_state"] == "stop":
        # Static circle when stopped
        st.markdown("""
        <div class='breathing-circle' style='background-color: #AAAAAA;'>
            Start
        </div>
        """, unsafe_allow_html=True)

# --- 8. CBT Thought Record ---
def cbt_thought_record_page():
    st.title("CBT Thought Record ✍️")
    st.markdown("Challenge and reframe unhelpful automatic negative thoughts (ANTs) using this 6-step cognitive restructuring tool.")
    
    st.markdown("---")
    st.subheader("Current Thought Record")

    # Use a form to capture all steps at once
    with st.form("cbt_record_form"):
        # Step 1-4: User input for the problem and initial evidence
        for i in range(4):
            st.markdown(CBT_PROMPTS[i])
            st.session_state["cbt_thought_record"][i] = st.text_area(
                "Your entry:",
                value=st.session_state["cbt_thought_record"].get(i, ""),
                key=f"cbt_input_{i}",
                placeholder="Type here...",
                height=50 if i < 3 else 150
            )

        # Step 6: User-input final Balanced Reframe (The AI step 5 is generated on submission)
        st.markdown(CBT_PROMPTS[5]) 
        st.session_state["cbt_thought_record"][5] = st.text_area(
            "Your entry:",
            value=st.session_state["cbt_thought_record"].get(5, ""),
            key=f"cbt_input_5",
            placeholder="Based on the evidence, what is a more balanced and helpful thought?",
            height=100
        )
        
        submitted = st.form_submit_button("Analyze & Save Record (Calls AI)")

        if submitted:
            # save_cbt_record handles the AI call for Step 5 and updates state
            if save_cbt_record(st.session_state["cbt_thought_record"]):
                st.rerun() # Rerun to display the AI analysis below

    st.markdown("---")

    # --- Display AI Reframing Card ---
    last_card = st.session_state.get("last_reframing_card")
    if last_card:
        st.subheader("🧠 Reframing Assistant's Analysis")
        
        with st.container(border=True):
            st.markdown(f"**Date:** {pd.to_datetime(last_card['date']).strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Situation:** {last_card['situation']}")
            st.markdown(f"**Emotion:** {last_card['emotion']}")
            st.markdown(f"**Negative Thought (ANT):** *“{last_card['thought']}”*")
            st.markdown("---")
            
            col_user, col_ai = st.columns(2)
            
            with col_user:
                st.markdown("#### User's Evidence FOR the Thought (Step 4)")
                st.markdown(last_card['evidence_for'])
                
            with col_ai:
                st.markdown("#### AI Assistant's Evidence AGAINST the Thought (Step 5)")
                st.markdown(last_card['ai_reframing'])
                
            st.markdown("---")
            st.success(f"**✅ Your Balanced Reframe (Step 6):** {last_card['balanced_reframe']}")
            
    st.markdown("---")

    # --- Display History ---
    st.subheader("Past Thought Records")
    if st.session_state.get("cbt_history"):
        for record in st.session_state["cbt_history"][1:]: # Skip the first (just displayed)
            with st.expander(f"Record from {pd.to_datetime(record['date']).strftime('%Y-%m-%d %H:%M')}: {record['emotion']}"):
                st.markdown(f"**Negative Thought:** *{record['thought']}*")
                st.success(f"**Balanced Reframe:** {record['balanced_reframe']}")
    else:
        st.info("No past thought records found.")

# --- 9. Journal Analysis (NEW FUNCTION) ---
def journal_analysis_page():
    st.title("Journal Analysis & Insights 📊")
    st.markdown("Review trends in your writing, emotions, and topics over time.")
    st.markdown("---")
    
    if not st.session_state.get("daily_journal"):
        st.info("You need at least a few journal entries to start the analysis. Try writing a quick entry on the Mindful Journaling page!")
        return

    df_journal = pd.DataFrame(st.session_state["daily_journal"])
    df_journal['date'] = pd.to_datetime(df_journal['date']).dt.date
    df_journal['sentiment'] = df_journal['sentiment'].astype(float)
    
    # Aggregate sentiment by date (in case of multiple entries per day)
    df_daily = df_journal.groupby('date').agg(
        avg_sentiment=('sentiment', 'mean'),
        entry_count=('text', 'size')
    ).reset_index()

    # --- 1. Sentiment Trend Chart ---
    st.subheader("Sentiment Score Trend")
    fig_sentiment = px.line(
        df_daily,
        x='date',
        y='avg_sentiment',
        title='Average Daily Journal Sentiment',
        markers=True,
        line_shape='spline',
        height=400
    )
    fig_sentiment.update_layout(
        yaxis=dict(range=[-1.1, 1.1], title="Sentiment Score", tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"]),
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.markdown("---")

    # --- 2. Entry Frequency ---
    st.subheader("Journaling Consistency")
    
    col_count, col_freq = st.columns(2)
    with col_count:
        st.metric("Total Entries", len(df_journal))
    with col_freq:
        st.metric("Total Days Logged", len(df_daily))

    fig_hist = px.bar(
        df_daily.tail(30), # Show last 30 days
        x='date',
        y='entry_count',
        title='Entries Logged Per Day (Last 30 Days)',
        height=300
    )
    fig_hist.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Entries",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")

    # --- 3. AI-Powered Summary ---
    st.subheader("AI Reflection on Your Themes 🧠")
    
    # Combine the text for AI summary (e.g., last 5 entries)
    # Ensure there are entries before accessing the list
    if st.session_state.get("daily_journal"):
        recent_entries = " | ".join([e['text'] for e in st.session_state["daily_journal"][:5]])
    else:
        recent_entries = ""
    
    if recent_entries:
        ai_prompt = f"""
        Analyze the following recent journal entries (separated by '|'): 
        "{recent_entries}"
        
        Act as a supportive therapist. Identify the **1-2 core emotional themes or life topics** discussed (e.g., school stress, family conflicts, self-doubt). 
        Then, give one sentence of empathetic validation and one small, actionable suggestion based on those themes. 
        Format your response with the themes bolded and suggestions bulleted. Keep the entire response under 5 sentences.
        """
        
        # Use a deterministic key to avoid re-running AI on every widget interaction
        ai_reflection_key = f"ai_reflection_journal_{hash(recent_entries)}"
        
        if ai_reflection_key not in st.session_state:
            with st.spinner("Generating personalized insights..."):
                reflection = safe_generate(ai_prompt, max_tokens=350)
            st.session_state[ai_reflection_key] = reflection
        else:
            reflection = st.session_state[ai_reflection_key]
            
        st.markdown(f"<div class='card' style='border-left: 5px solid #5D54A4;'>{reflection}</div>", unsafe_allow_html=True)
    else:
        st.info("Write a few more entries to unlock your personalized AI theme analysis.")

# --- 10. IoT Dashboard (ECE) ---
def iot_dashboard_page():
    st.title("IoT Dashboard (Simulated ECE Sensor Data) ⚙️")
    st.markdown("Real-time simulation of PPG (Heart Rate) and GSR (Stress) sensors, demonstrating Kalman Filter noise reduction.")
    st.markdown("---")

    # Initialize placeholders for the live data update
    if "live_chart_data" not in st.session_state:
        st.session_state["live_chart_data"] = pd.DataFrame(columns=["time_s", "Raw PPG", "Filtered HR", "GSR Stress"])

    # ECE data metrics (Heart Rate and Stress)
    col_hr, col_stress, col_mood = st.columns(3)
    
    hr_metric = col_hr.empty()
    stress_metric = col_stress.empty()
    mood_metric = col_mood.empty()

    # Placeholders for the live charts
    st.subheader("Live Physiological Signal Monitoring")
    hr_chart_placeholder = st.empty()
    
    st.subheader("GSR Stress Level Trend")
    stress_chart_placeholder = st.empty()

    # --- SIMULATION LOOP ---
    if st.button("Start/Stop Real-time Data Simulation", key="start_stop_ece_sim"):
        st.session_state["ece_running"] = not st.session_state.get("ece_running", False)
        
        if st.session_state["ece_running"]:
            st.toast("Simulation started!", icon="🟢")
        else:
            st.toast("Simulation stopped.", icon="🔴")
            
    if not st.session_state.get("ece_running", False):
        st.info("Press 'Start Simulation' to see the real-time data and Kalman filter in action.")
        return

    # Set up the chart configuration once
    chart_config = {
        'Filtered HR': {'color': '#5D54A4', 'name': 'Kalman Filtered HR (BPM)'},
        'Raw PPG': {'color': '#C7C4E1', 'name': 'Raw PPG (Noisy Signal)'}
    }

    # Start the real-time update loop
    while st.session_state.get("ece_running", False):
        current_time_ms = int(time.time() * 1000)
        
        # 1. Generate new simulated data point
        data_point = generate_simulated_physiological_data(current_time_ms)
        raw_ppg = data_point["raw_ppg_signal"]
        gsr_level = data_point["gsr_stress_level"]
        
        # 2. Apply Kalman Filter to the noisy HR measurement (raw_ppg is the measurement)
        # Note: kalman_filter_simple now handles the case where st.session_state["kalman_state"] is None
        filtered_hr, st.session_state["kalman_state"] = kalman_filter_simple(
            raw_ppg, 
            st.session_state.get("kalman_state")
        )
        
        # 3. Prepare data for plotting
        new_row = pd.DataFrame([{
            "time_s": (current_time_ms / 1000), 
            "Raw PPG": raw_ppg, 
            "Filtered HR": filtered_hr,
            "GSR Stress": gsr_level
        }])

        # Update the live chart data buffer (keep only the last 100 points)
        st.session_state["live_chart_data"] = pd.concat([st.session_state["live_chart_data"], new_row], ignore_index=True).tail(100)
        df_live = st.session_state["live_chart_data"]
        
        # 4. Update the Metrics
        hr_metric.metric("Heart Rate (BPM)", f"{filtered_hr:.1f}", delta=f"{filtered_hr - st.session_state['latest_ece_data'].get('filtered_hr', 75.0):.1f}")
        stress_metric.metric("GSR Stress Level", f"{gsr_level:.2f}")
        
        # Correlate stress with last logged mood
        mood_text = MOOD_EMOJI_MAP.get(int(st.session_state["mood_history"][0]["mood"])) if st.session_state.get("mood_history") and st.session_state["mood_history"] and st.session_state["mood_history"][0].get("mood") else "N/A"
        mood_metric.metric("Last Logged Mood", mood_text)
        
        st.session_state["latest_ece_data"] = {"filtered_hr": filtered_hr, "gsr_stress_level": gsr_level}

        # 5. Update the Heart Rate Chart (Raw vs Filtered)
        fig_hr = px.line(
            df_live, 
            x='time_s', 
            y=['Raw PPG', 'Filtered HR'],
            title='Heart Rate (BPM) - Raw vs. Kalman Filtered',
            color_discrete_map={'Raw PPG': '#C7C4E1', 'Filtered HR': '#5D54A4'},
            height=350
        )
        fig_hr.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Heart Rate (BPM)",
            legend_title="",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        hr_chart_placeholder.plotly_chart(fig_hr, use_container_width=True)

        # 6. Update the GSR Stress Chart
        fig_gsr = px.line(
            df_live, 
            x='time_s', 
            y='GSR Stress',
            title='GSR Stress Level Trend',
            color_discrete_sequence=['#DC3545'],
            height=300
        )
        fig_gsr.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="GSR Level (Relative)",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        stress_chart_placeholder.plotly_chart(fig_gsr, use_container_width=True)

        # 7. Sleep for a short interval (e.g., 0.5 seconds)
        time.sleep(0.5)

    # Simulation is stopped, display the final charts statically
    if st.session_state["live_chart_data"].empty:
        st.warning("No live data captured. Start the simulation to populate the charts.")
    else:
        st.info("Simulation is stopped. Press the button to resume.")

# --- 11. Report & Summary ---
def report_summary_page():
    st.title("My Wellness Report & Summary 📄")
    st.markdown("A non-interactive summary of your key wellness metrics and progress.")
    st.markdown("---")

    # 1. User & Date Info
    st.subheader(f"Report for {st.session_state.get('user_email', 'User')} | Generated: {datetime.now().strftime('%Y-%m-%d')}")
    st.markdown("---")

    # --- Mood and Engagement Summary ---
    st.subheader("1. Emotional & Engagement Overview")
    
    # Prepare Mood Data
    if st.session_state.get("mood_history"):
        df_mood = pd.DataFrame(st.session_state["mood_history"])
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.date
        avg_mood = df_mood['mood'].mean()
        mood_streak = st.session_state["streaks"]["mood_log"]
        
        mood_emoji_display = MOOD_EMOJI_MAP.get(int(round(avg_mood)), "❓")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Avg Mood Score", f"{avg_mood:.1f} / 11", help="Average of all logged moods.")
        with col_b:
            st.metric("Mood Streak", f"{mood_streak} days")
        with col_c:
            st.metric("Most Common Mood", mood_emoji_display.split(" ")[1])

        st.markdown("**Mood Distribution (Last 30 Logs):**")
        mood_counts = df_mood.head(30)['mood'].value_counts().sort_index().reset_index()
        mood_counts.columns = ['Mood Score', 'Count']
        mood_counts['Mood Label'] = mood_counts['Mood Score'].apply(lambda x: MOOD_EMOJI_MAP.get(x, f"{x}"))
        
        fig_mood = px.bar(
            mood_counts,
            x='Mood Label',
            y='Count',
            title='Frequency of Mood Scores',
            color='Count',
            color_continuous_scale=px.colors.sequential.Purp,
            height=300
        )
        fig_mood.update_layout(xaxis_title="", yaxis_title="Number of Logs", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mood, use_container_width=True)
    else:
        st.info("No mood data to generate summary charts.")
        
    st.markdown("---")

    # --- Wellness Check-in Summary ---
    st.subheader("2. Wellness Check-in (PHQ-9) Status")
    
    phq9_score_display = st.session_state.get("phq9_score")
    if phq9_score_display is not None:
        phq9_interpretation = st.session_state.get("phq9_interpretation", "N/A")
        last_phq9_date = st.session_state.get("last_phq9_date", "N/A")
        
        col_d, col_e = st.columns(2)
        with col_d:
            st.metric("Latest Score", phq9_score_display)
        with col_e:
            st.metric("Interpretation", phq9_interpretation)
            
        st.caption(f"Last taken on: {last_phq9_date}")
        
        if phq9_score_display is not None and phq9_score_display >= 10:
             st.warning(f"Note: Your latest score of {phq9_score_display} suggests **{phq9_interpretation}** severity. Consistent use of journaling and CBT is recommended.")
    else:
        st.info("No Wellness Check-in data (PHQ-9) found.")
        
    st.markdown("---")
    
    # --- Journaling Summary ---
    st.subheader("3. Journaling & Reflection")
    if st.session_state.get("daily_journal"):
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        avg_sentiment = df_journal['sentiment'].mean()
        
        sentiment_text = "Positive" if avg_sentiment > 0.05 else ("Negative" if avg_sentiment < -0.05 else "Neutral")
        
        col_f, col_g = st.columns(2)
        with col_f:
            st.metric("Total Entries", len(df_journal))
        with col_g:
            st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f} ({sentiment_text})")
            
        # Display the most negative and most positive entry
        most_pos_entry = df_journal.loc[df_journal['sentiment'].idxmax()]
        most_neg_entry = df_journal.loc[df_journal['sentiment'].idxmin()]
        
        st.markdown(f"**Most Positive Entry ({most_pos_entry['sentiment']:.2f}):** *{most_pos_entry['text'][:100]}...*")
        st.markdown(f"**Most Negative Entry ({most_neg_entry['sentiment']:.2f}):** *{most_neg_entry['text'][:100]}...*")
        
    else:
        st.info("No journal entries found.")
        
    st.markdown("---")
    
    # --- CBT Summary ---
    st.subheader("4. Cognitive Behavioral Therapy (CBT) Use")
    cbt_count = len(st.session_state.get("cbt_history", []))
    st.metric("Total Thought Records Completed", cbt_count)
    
    if cbt_count > 0:
        st.markdown("**Last Reframed Thought:**")
        last_cbt = st.session_state["cbt_history"][0]
        st.markdown(f"**Negative Thought:** *{last_cbt['thought']}*")
        st.success(f"**Balanced Reframe:** {last_cbt['balanced_reframe']}")
    else:
        st.info("No CBT Thought Records completed yet.")

# --- 12. Generic Placeholder for unbuilt pages ---
def generic_placeholder_page(page_name):
    """
    Standard placeholder for unbuilt pages. 
    It avoids the 'Coming Soon' warning and provides a clean header.
    """
    st.title(page_name)
    st.markdown("---")
    st.info(f"This is the dedicated page for the **{page_name}** feature. It's ready for development!")
    # FIX: Use markdown div to prevent Streamlit MediaFileStorageError on placeholder
    st.markdown("<div style='background-color: #e0e0e0; height: 300px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; color: #555;'>[Placeholder: Image of under construction sign]</div>", unsafe_allow_html=True)
    st.caption("Feature development is underway.")


# --- MAIN PAGE ROUTER (FINAL VERSION - CORRECTED IMAGE SOURCE) ---
if st.session_state.get("logged_in") is False:
    # --- LOGOUT/WELCOME PAGE (FIXED IMAGE SOURCE ERROR) ---
    st.title("Youth Wellness App")
    st.markdown("---")
    st.info("Please use the sidebar to log in or register to access the dashboard features.")
    # FIX: Use markdown div to prevent Streamlit MediaFileStorageError
    st.markdown("<div style='background-color: #d1d8e0; height: 400px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #333;'>[Placeholder: Image of wellness tools]</div>", unsafe_allow_html=True)
    st.caption("Wellness is a journey.")

else:
    # --- AUTHENTICATED PAGES ---
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
    elif current_page == "Mindful Breathing":
        mindful_breathing_page()
    elif current_page == "CBT Thought Record":
        cbt_thought_record_page()
    elif current_page == "Journal Analysis":
        journal_analysis_page()
    elif current_page == "IoT Dashboard (ECE)": 
        iot_dashboard_page()
    elif current_page == "Report & Summary": 
        report_summary_page()

    # 2. All Other Placeholder Pages (Should be empty now!)
    else:
        # Fallback in case a key is accidentally missed, but should not be hit.
        generic_placeholder_page(current_page)