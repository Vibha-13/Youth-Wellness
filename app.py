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
    # Check if the user is logged in
    is_logged_in = st.session_state.get("logged_in", False)
    
    # --- CSS STYLING ---
    st.markdown(f"""
<style>
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
textarea, input[type="text"], input[type="email"] {{
    color: #1E1E1E !important;
    -webkit-text-fill-color: #1E1E1E !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border: 2px solid #FFD6E0 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}}
textarea:focus, input[type="text"]:focus, input[type="email"]:focus {{
    border-color: #FF9CC2 !important;
    box-shadow: 0 0 8px rgba(255, 156, 194, 0.5);
}}

/* 3. Custom Card Style (Wellness Look) */
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

/* 4. Sidebar Styles (HIDES when NOT logged in for clean transition) */
[data-testid="stSidebar"] {{
    background: linear-gradient(to bottom, #fff0f5, #e0f7fa);
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    transition: transform 0.3s ease-in-out;
    /* CRITICAL: Hide when not logged in or during splash */
    {'visibility: hidden; transform: translateX(-100%); width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}
/* Ensures the sidebar is completely gone */
[data-testid="stSidebar"] > div:first-child {{
    {'width: 0 !important;' if not (is_logged_in and not st.session_state.get("show_splash")) else ''}
}}


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
.status-connected {{ background-color: #D4EDDA; color: #155724; border-left: 4px solid #28A745; }}
.status-local {{ background-color: #FFF3CD; color: #856404; border-left: 4px solid #FFC107; }}

/* 7. Hide Streamlit Footer */
footer {{
    visibility: hidden;
}}

/* Breathing Effect CSS (NEW VISUALS) */
@keyframes pulse-in {
    0% { transform: scale(0.6); opacity: 0.8; }
    100% { transform: scale(1.0); opacity: 1.0; }
}
@keyframes pulse-out {
    0% { transform: scale(1.0); opacity: 1.0; }
    100% { transform: scale(0.6); opacity: 0.8; }
}

.breathing-circle {
    width: 300px;
    height: 300px;
    background-color: #FF9CC2; /* Your primary theme color */
    border-radius: 50%;
    margin: 50px auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.5rem;
    font-weight: bold;
    box-shadow: 0 0 50px rgba(255, 156, 194, 0.7);
    transition: background-color 0.5s;
}

/* Specific state animations */
.inhale {
    animation: pulse-in 4s ease-in-out forwards; /* 4 seconds inhale */
    background-color: #FF9CC2; 
}
.hold {
    transform: scale(1.0); /* Keeps scale steady */
    animation-duration: 7s; /* 7 seconds hold (no animation, just steady state) */
    background-color: #FF6F91;
}
.exhale {
    animation: pulse-out 8s ease-in-out forwards; /* 8 seconds exhale */
    background-color: #6A8DFF;
}

</style>
""", unsafe_allow_html=True)

# Call the setup function early in the main script flow
setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------
@st.cache_data
def initialize_kalman(Q_val=0.01, R_val=0.1):
    """Initializes the Kalman filter state variables."""
    # Q: Process Noise Covariance (how much the state changes naturally)
    # R: Measurement Noise Covariance (how much the sensor is noisy)
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
        state = initialize_kalman() 
        st.session_state["kalman_state"] = state
    # ---------------------------------------------
    
    # 1. Prediction
    x_pred = state['x_est'] 
    P_pred = state['P_est'] + state['Q']

    # 2. Update (Calculate Kalman Gain K)
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
    # Uses a slow sine wave to simulate baseline changes
    base_hr = 85 + 10 * np.sin(time_sec / 30.0) 
    
    # Simulate Filtered HR (The 'clean' signal we *want* to see)
    clean_hr = base_hr + 2 * np.sin(time_sec / 0.5) 
    
    # GSR/Stress Simulation (correlated with base HR and overall phq9 score)
    phq9_score = st.session_state.get("phq9_score") or 0
    # Normalize score by max possible (27)
    gsr_base = 1.0 + (base_hr / 100.0) + 0.5 * (phq9_score / 27.0)
    gsr_noise = 0.5 * random.gauss(0, 1) # Add some noise to GSR
    gsr_value = gsr_base + gsr_noise
    
    # Add high-frequency noise for the raw PPG measurement
    ppg_noise = 3 * random.gauss(0, 1)
    raw_ppg_signal = clean_hr + ppg_noise
    
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
    # This sets up the *regular* RLS-secured client for normal operations (e.g., fetching, saving mood/journal)
    if not url or not key:
        return None, False
    try:
        if not url or not key:
             return None, False
        
        client = create_client(url, key)
        return client, True
    except Exception:
        return None, False
        
        
# --- CRITICAL: ADMIN CLIENT FOR REGISTRATION ---
@st.cache_resource(show_spinner=False)
def get_supabase_admin_client():
    """
    Initializes a Supabase client using the Service Role Key (Admin Key).
    This client is used for secure user registration, bypassing RLS.
    """
    try:
        # Load URL and SERVICE_KEY securely
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        # Use a distinct key for the Service Role Key (Admin Key)
        key = st.secrets.get("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY")) 
        
        if not url or not key:
            # Print to Streamlit logs to signal missing key
            print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY is missing/empty. Admin client cannot be initialized.")
            return None
        
        # Ensure the keys are stripped of any surrounding quotes or whitespace
        url_clean = url.strip().strip('"') if isinstance(url, str) else None
        key_clean = key.strip().strip('"') if isinstance(key, str) else None
        
        if not url_clean or not key_clean:
            print("ERROR: SUPABASE credentials failed cleaning check.")
            return None
            
        return create_client(url_clean, key_clean)
    except Exception as e:
        # Print actual error for debugging
        print(f"ERROR initializing Supabase Admin Client: {e}")
        return None


# ---------- Session state defaults (CLEANED UP) ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# [NEW TRANSITION STATE]
if "show_splash" not in st.session_state:
    st.session_state["show_splash"] = True 

# IoT/ECE State
if "kalman_state" not in st.session_state:
    st.session_state["kalman_state"] = initialize_kalman()
# --- ADDED kalman_hr to physiological_data DataFrame ---
if "physiological_data" not in st.session_state:
    st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level", "kalman_hr"])
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

# ---------- AI/Sentiment Helper functions (All preserved) ----------
def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300):
    """
    Generate text via OpenRouter, with system message and custom fallback.
    (Logic preserved from previous version)
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

        is_chat_context = st.session_state["page"] == "AI Chat"
        if is_chat_context and messages_for_api and (messages_for_api[-1]["content"] != prompt_clean or messages_for_api[-1]["role"] != "user"):
            messages_for_api.append({"role": "user", "content": prompt_clean})
        elif not is_chat_context:
            system_prompt = st.session_state.chat_messages[0]["content"] if st.session_state.chat_messages and st.session_state.chat_messages[0]["role"] == "system" else "You are a helpful AI assistant."
            messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_clean}]


        try:
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

# ---------- Supabase helpers (DB functions) ----------

def register_user_db(email: str):
    """
    Inserts a new user entry into the 'users' and 'profiles' tables 
    using the dedicated Admin Client to bypass RLS.
    """
    # Retrieve the ADMIN client (guaranteed to be initialized correctly)
    admin_client = get_supabase_admin_client()
    
    # Check if the client initialization failed (e.g., key is missing)
    if not admin_client:
        st.error("Admin Client Initialization Error: Please ensure `SUPABASE_SERVICE_KEY` and `SUPABASE_URL` are set correctly in your Streamlit secrets file.")
        return None 
        
    # 1. Generate a valid UUID for the new user ID
    new_user_id = str(uuid.uuid4())
    
    # 2. Get current timestamp in ISO format for PostgreSQL
    current_time = datetime.now().isoformat() 
    
    try:
        # 3. Insert into 'users' table 
        admin_client.table("users").insert({
            "id": new_user_id,
            "email": email,
            "created_at": current_time 
        }).execute()

        # 4. Also insert into 'profiles' table 
        admin_client.table("profiles").insert({
            "id": new_user_id,
            "created_at": current_time
        }).execute()
        
        # If both inserts succeed, the function returns the ID
        return new_user_id
            
    except Exception as e:
        st.error(f"DB Insert Error: {e}") 
        return None

def get_user_by_email_db(email: str):
    """
    Searches the database for an existing user's ID using their email.
    Uses the ADMIN CLIENT to bypass RLS, ensuring a reliable lookup.
    """
    # CRITICAL FIX: Use the RLS-bypassing Admin Client for lookup
    supabase_client = get_supabase_admin_client()
    
    if not supabase_client:
        # If the Admin Client setup failed (e.g., Service Key missing), return empty
        return []
        
    try:
        # Query the 'users' table (confirmed to hold the email constraint)
        # Using the admin client ensures this select statement executes successfully.
        res = supabase_client.table("users").select("id, email").eq("email", email).execute()
        
        # This will return the user data if found, or an empty list if not.
        return res.data or []

    except Exception as e:
        # If the lookup fails here, the Supabase URL or Service Key is highly likely incorrect.
        st.error(f"CRITICAL ADMIN LOOKUP FAIL: {e}") 
        return []


# --- SAVE FUNCTIONS (Preserved) ---
def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        st.session_state["daily_goals"]["journal_entry"]["count"] += 1
        return True
    except Exception:
        return False

def save_mood_db(user_id, mood: int, note: str) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("mood_logs").insert({"user_id": user_id, "mood_score": mood, "note": note}).execute()
        st.session_state["daily_goals"]["log_mood"]["count"] += 1
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
        
def save_cbt_record(cbt_data: dict):
    # Logic preserved
    supabase_client = st.session_state.get("_supabase_client_obj")
    user_id = st.session_state.get("user_id")

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
        "user_id": user_id,
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
    
    # Optional: Save to DB
    if supabase_client:
        try:
            supabase_client.table("cbt_records").insert(record).execute()
        except Exception:
            st.warning("Could not save CBT record to database.")
            
    return True

# --- BEGIN CACHE ERROR FIX ---
@st.cache_data(show_spinner=False)
def load_all_user_data(user_id): # FIX 1: Removed 'supabase_client' argument
    """
    Loads all user-specific wellness data from the database.
    
    The Supabase client is fetched internally from st.session_state to ensure 
    only the hashable 'user_id' is passed to this cached function, avoiding the 
    UnhashableParamError.
    """
    # FIX 2: Retrieve the client internally
    supabase_client = st.session_state.get("_supabase_client_obj")

    if not supabase_client:
        return {"journal": [], "mood": [], "phq9": [], "ece": [], "cbt": []}
    
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

        # Load CBT History
        res_c = supabase_client.table("cbt_records").select("*").eq("user_id", user_id).order("date", desc=True).execute()
        data["cbt"] = res_c.data or []


    except Exception as e:
        return {"journal": [], "mood": [], "phq9": [], "ece": [], "cbt": []}
        
    return data
# --- END CACHE ERROR FIX ---

def calculate_plant_health():
    # Logic preserved
    health_base = 50.0 
    
    goals = st.session_state.get("daily_goals")
    if goals is None:
        goals = DEFAULT_GOALS.copy()
        st.session_state["daily_goals"] = goals

    goal_completion_score = 0
    total_goals = len(goals)
    if total_goals > 0:
        for goal_key, goal in goals.items():
            if isinstance(goal, dict) and goal.get("count", 0) >= goal.get("target", 1):
                goal_completion_score += 1
        
        health_base += (goal_completion_score / total_goals) * 30.0

    if st.session_state["mood_history"]:
        df_mood = pd.DataFrame(st.session_state["mood_history"]).head(7) 
        if not df_mood.empty:
            df_mood['mood'] = pd.to_numeric(df_mood['mood'], errors='coerce')
            avg_mood = df_mood['mood'].mean()
            # Normalize mood (6 is neutral, so anything above 6 helps)
            mood_contribution = (avg_mood - 6.0) * 4 
            health_base += mood_contribution

    st.session_state["plant_health"] = max(0, min(100, health_base))
    
# !!! FIX APPLIED HERE: Added type check for robustness !!!
def check_and_reset_goals():
    today = datetime.now().date()
    
    if st.session_state.get("daily_goals") is None:
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()

    goals = st.session_state["daily_goals"]
    
    for key, goal in goals.items():
        # CRITICAL FIX: Ensure 'goal' is a dictionary before accessing attributes
        if not isinstance(goal, dict):
            # If corrupted, reset just this entry to the default structure
            goals[key] = DEFAULT_GOALS.get(key, {"count": 0, "target": 1, "last_reset": None})
            goal = goals[key] # Update the reference
            
        last_reset = goal.get("last_reset")
        if last_reset:
            try:
                last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d").date()
            except ValueError:
                # Fallback if the date format is wrong
                last_reset_date = today - timedelta(days=1) 
                
            if last_reset_date < today:
                goal["count"] = 0
                goal["last_reset"] = today.strftime("%Y-%m-%d")
        elif last_reset is None:
            goal["last_reset"] = today.strftime("%Y-%m-%d")

    st.session_state["daily_goals"] = goals
    calculate_plant_health() 

# Run goal check on every app load
check_and_reset_goals()

# ---------- PAGE CONTENT FUNCTIONS (Full Implementation for Key Features) ----------

# [NEW] Splash Screen Function for Initial Transition
def app_splash_screen():
    # Use a container to center the content vertically and horizontally
    col_a, col_b, col_c = st.columns([1, 4, 1])

    with col_b:
        # Custom HTML/CSS for a large, centered title
        st.markdown("""
        <div style="text-align: center; margin-top: 20vh; animation: fadeIn 2s ease-in-out;">
            <h1 style="font-size: 5rem; color: #FF9CC2; margin-bottom: 0; animation: bounce 1.5s infinite alternate;">HarmonySphere</h1>
            <p style="font-size: 1.5rem; color: #555;">Your Youth Wellness Companion</p>
        </div>
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounce {
            0% { transform: translateY(0); }
            100% { transform: translateY(-10px); }
        }
        </style>
        """, unsafe_allow_html=True)
    # Use a slightly longer delay (2.0s) to create a more noticeable transition effect
    if st.session_state["show_splash"]:
        # Increased delay for better visibility
        time.sleep(2.0)
        st.session_state["show_splash"] = False
        # Crucial: Rerun to proceed to the next stage (unauthenticated_home)
        st.rerun()

# !!! FIX APPLIED HERE: Restructured for centered, unauthenticated login !!!
def unauthenticated_home():
    # Use a container to center the content
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.title("Welcome to HarmonySphere 🧠")
    st.subheader(random.choice(QUOTES))
    st.markdown("</div>", unsafe_allow_html=True)

    # Center the login form using columns
    col_a, col_form, col_b = st.columns([1.5, 2, 1.5])
    
    with col_form:
        # Custom HTML styling for the white box and shadow
        st.markdown("""
        <div style="background-color: white; padding: 30px; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.15);">
            <h3 style="text-align: center; color: #FF9CC2; margin-top: 0;">Access Your Wellness Dashboard</h3>
            <p style="text-align: center; font-size: 0.9rem; color: #555;">Use your email to securely log in or register.</p>
        """, unsafe_allow_html=True)
        
        with st.form("centered_login_form"):
            email = st.text_input("Email", placeholder="teenager@example.com", key="login_email_center").lower().strip()
            submitted = st.form_submit_button("Access Dashboard", use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # IMPORTANT: Authentication Logic runs ONLY if submitted
    if submitted:
        if email and "@" in email:
            # Clear existing session data before logging in
            # This logic is kept for robust multi-user capability
            for key in ["user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state", "daily_journal", "mood_history", "physiological_data", "ece_history", "plant_health", "cbt_history", "last_reframing_card"]:
                if key in st.session_state:
                    if key in ["user_id", "user_email", "phq9_score", "phq9_interpretation"]:
                        st.session_state[key] = None
                    elif key in ["kalman_state"]:
                        st.session_state[key] = initialize_kalman()
                    elif key in ["daily_journal", "mood_history", "ece_history", "cbt_history"]:
                        st.session_state[key] = []
                    elif key in ["physiological_data"]:
                        # Ensure the DataFrame resets to the correct columns
                        st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level", "kalman_hr"])
                    elif key in ["plant_health"]:
                        st.session_state[key] = 70.0
                    elif key in ["last_reframing_card"]:
                        st.session_state[key] = None

            user = None
            db_connected = st.session_state.get("_db_connected")

            # --- 1. Login/Lookup Attempt ---
            if db_connected:
                user_list = get_user_by_email_db(email)
                if user_list:
                    user = user_list[0]

            if user or db_connected is False:
                # --- AUTHENTICATION SUCCESS (Existing User or Local Mode) ---
                st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                st.session_state["user_email"] = email
                st.session_state["logged_in"] = True

                # --- DATA LOADING (Transition Start) ---
                if user and db_connected:
                    with st.spinner("Loading your personalized wellness data..."):
                        # FIX 3: Corrected function call - only passing user_id
                        user_data = load_all_user_data(st.session_state["user_id"]) 

                        st.session_state["daily_journal"] = user_data["journal"]
                        st.session_state["mood_history"] = user_data["mood"]
                        st.session_state["ece_history"] = user_data["ece"]
                        st.session_state["cbt_history"] = user_data["cbt"]
                        
                        if user_data["phq9"]:
                            latest_phq9 = user_data["phq9"][0]
                            st.session_state["phq9_score"] = latest_phq9.get("score")
                            st.session_state["phq9_interpretation"] = latest_phq9.get("interpretation")
                            st.session_state["last_phq9_date"] = pd.to_datetime(latest_phq9.get("created_at")).strftime("%Y-%m-%d")

                # The smooth transition effect you asked for!
                st.success("Login successful! Redirecting to dashboard...")
                time.sleep(1.0)
                st.session_state["page"] = "Home"
                st.rerun()

            else: 
                # --- 2. Registration (New User) ---
                if db_connected:
                    if st.button(f"Register as New User: {email}", use_container_width=True):
                        new_id = register_user_db(email)
                        if new_id:
                            st.session_state["user_id"] = new_id
                            st.session_state["user_email"] = email
                            st.session_state["logged_in"] = True
                            st.success("Registration successful! Welcome to HarmonySphere.")
                            time.sleep(1.0)
                            st.session_state["page"] = "Home"
                            st.rerun()
                        else:
                            st.error("Registration failed. Check database logs or connection.")
                else:
                    st.error("Cannot connect to the database. Running in local mode only.")
                    
        else:
            st.error("Please enter a valid email address.")
            

def sidebar_status_display():
    # Logic preserved
    
    # --- 1. Connection Status ---
    db_connected = st.session_state.get("_db_connected", False)
    ai_available = st.session_state.get("_ai_available", False)
    logged_in = st.session_state.get("logged_in", False)
    
    if logged_in:
        st.markdown(f"**Logged In as:** `{st.session_state.get('user_email', 'Local User')}`")
    else:
        st.markdown("**Not Logged In**")
        
    st.divider()

    st.markdown(f"""
    <div class='sidebar-status {'status-connected' if db_connected else 'status-local'}'>
        Database: {'Connected (Supabase)' if db_connected else 'Local Mode (No Save)'}
    </div>
    <div class='sidebar-status {'status-connected' if ai_available else 'status-local'}'>
        AI Model: {'Connected (OpenRouter)' if ai_available else 'Disabled (Missing Key)'}
    </div>
    """, unsafe_allow_html=True)


def sidebar_navigation():
    # Logic preserved
    st.sidebar.markdown(f"## 🌿 HarmonySphere")
    
    # Check if the user is logged in before showing nav
    if not st.session_state.get("logged_in"):
        return

    # User Info and Health
    sidebar_status_display()
    
    # Current Health Widget (Plant Gamification)
    st.sidebar.markdown("### Your Wellness Plant")
    plant_health = st.session_state.get("plant_health", 70.0)
    
    # Dynamic Plant Emoji and Message
    if plant_health > 85:
        emoji = "🌳"
        msg = "Thriving! Keep up the great work."
    elif plant_health > 50:
        emoji = "🌱"
        msg = "Healthy and growing. Water regularly!"
    elif plant_health > 25:
        emoji = "😟"
        msg = "A bit droopy. Focus on your goals today."
    else:
        emoji = "🚨"
        msg = "Needs immediate care! Check in with yourself."

    st.sidebar.markdown(f"<h1 style='text-align:center; font-size: 3rem;'>{emoji}</h1>", unsafe_allow_html=True)
    st.sidebar.progress(int(plant_health))
    st.sidebar.markdown(f"<p style='text-align:center; font-size: 0.9rem;'>{msg}</p>", unsafe_allow_html=True)
    
    st.sidebar.divider()
    
    # PHQ-9 Status Display
    phq9_score = st.session_state.get("phq9_score")
    last_phq9_date = st.session_state.get("last_phq9_date")
    
    if phq9_score is not None:
        st.sidebar.markdown(f"""
            <p style='font-size: 0.9rem; font-weight: 600; color: #444;'>
            Latest PHQ-9 Score: 
            <span style='color: #FF6F91; font-size: 1.1rem;'>
            {phq9_score}/27 
            </span> 
            <br>
            <span style='font-weight: 400; font-size: 0.8rem; color: #777;'>
            (Last check: {last_phq9_date})
            </span>
            </p>
        """, unsafe_allow_html=True)
    else:
         st.sidebar.warning("Complete a Wellness Check-in!")
    
    st.sidebar.divider()
    
    # Navigation Buttons
    pages = [
        "Home", 
        "Mindful Journaling", 
        "Mood Tracker", 
        "CBT Thought Record",
        "AI Chat", 
        "Wellness Check-in",
        "Mindful Breathing",
        "IoT Dashboard (ECE)",
        "Report & Summary"
    ]
    
    for page in pages:
        if st.sidebar.button(page, use_container_width=True, key=f"nav_{page}"):
            st.session_state["page"] = page
            # Reset CBT form when navigating away
            if page != "CBT Thought Record":
                st.session_state["cbt_thought_record"] = {i: "" for i in range(len(CBT_PROMPTS))}
            st.rerun()

    st.sidebar.divider()
    
    # Logout button
    if st.sidebar.button("Logout", use_container_width=True):
        # Clear all user-specific state upon logout
        keys_to_clear = [
            "logged_in", "user_id", "user_email", "phq9_score", 
            "phq9_interpretation", "last_phq9_date", "daily_journal", 
            "mood_history", "ece_history", "cbt_history", "last_reframing_card",
            "plant_health", "daily_goals"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset complex state objects separately
        st.session_state["kalman_state"] = initialize_kalman()
        st.session_state["physiological_data"] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level", "kalman_hr"])
        st.session_state["daily_goals"] = DEFAULT_GOALS.copy() # Reset goals
        
        # Go back to splash/unauthenticated home
        st.session_state["show_splash"] = True
        st.session_state["page"] = "Home"
        st.rerun()


def dashboard_metric(title, value, unit="", icon="✨", color="#FF9CC2"):
    # Logic preserved
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <p style="font-size: 1rem; color: #555; margin: 0;">{title}</p>
            <span style="font-size: 1.5rem; color: {color};">{icon}</span>
        </div>
        <h2 style="font-size: 2.5rem; color: {color}; margin-top: 5px; margin-bottom: 0;">
            {value} <span style="font-size: 1.2rem; color: #777;">{unit}</span>
        </h2>
    </div>
    """, unsafe_allow_html=True)


def homepage_panel():
    # Logic preserved
    st.title("Welcome Back, " + (st.session_state.get("user_email", "User").split('@')[0] if '@' in st.session_state.get("user_email", "User") else st.session_state.get("user_email", "User")) + "!")
    st.markdown("### Your Personalized Wellness Dashboard")
    st.markdown("---")
    
    # --- Row 1: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Mood Metric
    latest_mood = st.session_state["mood_history"][0] if st.session_state["mood_history"] else None
    if latest_mood:
        mood_emoji = MOOD_EMOJI_MAP.get(latest_mood["mood"], "🤔")
        mood_text = mood_emoji.split(" ")[1]
        dashboard_metric("Latest Mood", mood_text, icon=mood_emoji.split(" ")[0], color="#FF6F91")
    else:
        dashboard_metric("Latest Mood", "N/A", icon="🤔", color="#ccc")
        
    # Journal Streak Metric
    journal_days = len(st.session_state["daily_journal"])
    dashboard_metric("Total Journal Entries", str(journal_days), icon="📝", color="#6A8DFF")

    # PHQ-9 Score Metric
    phq9_score = st.session_state.get("phq9_score")
    phq9_text = str(phq9_score) if phq9_score is not None else "N/A"
    dashboard_metric("Latest PHQ-9 Score", phq9_text, "/27", icon="🧠", color="#28A745" if phq9_score is not None and phq9_score < 10 else "#FFC107")

    # HR/Stress Metric
    hr = st.session_state["latest_ece_data"].get("filtered_hr", 75.0)
    gsr = st.session_state["latest_ece_data"].get("gsr_stress_level", 1.0)
    # Simple color coding for stress (1.0 is low, 3.0 is high)
    stress_color = "#28A745" if gsr < 1.5 else "#FFC107" if gsr < 2.5 else "#FF6F91"
    dashboard_metric("Current Heart Rate", f"{hr:.0f}", "BPM", icon="❤️", color=stress_color)

    st.markdown("---")
    
    # --- Row 2: Goals & Quick Actions ---
    col_goals, col_actions = st.columns([1.5, 1])

    with col_goals:
        st.subheader("Today's Wellness Goals")
        goals = st.session_state.get("daily_goals", DEFAULT_GOALS)
        
        goal_keys = list(goals.keys())
        for key in goal_keys:
            goal = goals[key]
            progress = min(goal["count"] / goal["target"], 1.0)
            
            with st.container(border=True):
                col_g1, col_g2 = st.columns([3, 1])
                with col_g1:
                    st.markdown(f"**{goal['name']}** ({goal['count']}/{goal['target']})")
                    st.progress(progress)
                with col_g2:
                    if progress >= 1.0:
                        st.success("Done!")
                    elif st.button("Complete", key=f"goal_complete_{key}", use_container_width=True):
                        # Simple increment logic (goal is checked/reset in check_and_reset_goals())
                        goals[key]["count"] += 1
                        calculate_plant_health()
                        st.session_state["daily_goals"] = goals
                        st.toast(f"Goal '{goal['name']}' Completed! 🌱", icon="🎉")
                        time.sleep(0.5)
                        st.rerun()

    with col_actions:
        st.subheader("Quick Actions")
        if st.button("Log Today's Mood", use_container_width=True):
            st.session_state["page"] = "Mood Tracker"
            st.rerun()
        if st.button("Start AI Chat", use_container_width=True):
            st.session_state["page"] = "AI Chat"
            st.rerun()
        if st.button("Practice Breathing", use_container_width=True):
            st.session_state["page"] = "Mindful Breathing"
            st.rerun()

    st.markdown("---")
    
    # --- Row 3: Visualization (Mood Over Time) ---
    st.subheader("Mood Trends (Last 30 Logs)")
    
    mood_data = st.session_state["mood_history"]
    if mood_data:
        df_mood = pd.DataFrame(mood_data)
        # Convert to datetime and ensure the latest is at the bottom for chronological plotting
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        df_mood = df_mood.sort_values('date').head(30)
        df_mood['mood_label'] = df_mood['mood'].apply(lambda x: MOOD_EMOJI_MAP.get(x, "N/A"))
        
        fig = px.line(
            df_mood, 
            x='date', 
            y='mood', 
            markers=True, 
            height=300, 
            title='Mood Score Over Time (1=Worst, 10=Best)',
            template="plotly_white"
        )
        
        # Add labels to markers
        fig.update_traces(
            mode='lines+markers+text', 
            text=df_mood['mood_label'].str.split(" ", expand=True)[0], # Use only the emoji
            textposition='top center',
            marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey'))
        )
        
        # Set y-axis to be discrete and labeled
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=list(MOOD_EMOJI_MAP.keys())[:-1], # 1 to 10
                ticktext=[MOOD_EMOJI_MAP[i] for i in range(1, 11)],
                range=[0.5, 10.5] # Padding for the y-axis
            ),
            xaxis_title=None,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your mood a few times to see your personalized trends here!")

# -------------------- Journaling Page --------------------
def mindful_journaling_page():
    # Logic preserved
    st.title("📝 Mindful Journaling")
    st.subheader("Reflect on your day, process your feelings, and track your thoughts.")
    st.markdown("---")

    journal_text = st.text_area(
        "**What's on your mind right now?**", 
        height=250, 
        placeholder="Write a few sentences about your thoughts, feelings, or what happened today...",
        key="current_journal_entry"
    )

    if st.button("Save Entry and Analyze Sentiment", use_container_width=True):
        if len(journal_text.split()) < 5:
            st.warning("Please write at least 5 words for a meaningful entry.")
        else:
            sentiment = sentiment_compound(journal_text)
            user_id = st.session_state.get("user_id")
            
            if save_journal_db(user_id, journal_text, sentiment):
                
                # Update local state immediately
                new_entry = {
                    "date": datetime.now().isoformat(),
                    "text": journal_text,
                    "sentiment": sentiment
                }
                st.session_state["daily_journal"].insert(0, new_entry)
                
                st.success("Journal Entry Saved!")
                
                # Sentiment feedback
                if sentiment > 0.3:
                    feedback = "That's a very positive entry! Keep focusing on the good things. 😊"
                elif sentiment < -0.3:
                    feedback = "I hear some negativity here. It's okay to feel that way. Writing it down is the first step. Let's process it. 🫂"
                else:
                    feedback = "A thoughtful, balanced entry. Reflection is key to growth. 🌱"
                
                st.info(f"Sentiment Analysis Score: **{sentiment:.2f}**. {feedback}")
                
                # Clear the text area and rerun to update the list below
                st.session_state["current_journal_entry"] = ""
                st.rerun()
            else:
                st.error("Failed to save entry. Check database connection if running outside of local mode.")

    st.markdown("---")
    st.subheader("Your Recent Journal Entries")
    
    if st.session_state["daily_journal"]:
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        df_journal['date'] = pd.to_datetime(df_journal['date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display as expandable containers
        for index, row in df_journal.head(10).iterrows():
            sentiment_score = row['sentiment']
            
            # Simple color coding for the title/border
            if sentiment_score > 0.3:
                color = "#28A745"
            elif sentiment_score < -0.3:
                color = "#FF6F91"
            else:
                color = "#FFC107"
            
            with st.expander(f"**{row['date']}** | Sentiment: **{sentiment_score:.2f}**", expanded=False):
                st.markdown(f'<div style="border-left: 5px solid {color}; padding-left: 10px;">', unsafe_allow_html=True)
                st.markdown(row['text'])
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Your saved journal entries will appear here.")

# -------------------- Mood Tracker Page --------------------
def mood_tracker_page():
    # Logic preserved
    st.title("😊 Mood Tracker")
    st.subheader("Check in with your emotions now. Self-awareness is the first step.")
    st.markdown("---")

    col_slider, col_mood_info = st.columns([2, 1])

    with col_slider:
        mood_score = st.slider(
            "**How are you feeling right now?** (1 = Worst, 10 = Best)",
            min_value=1, 
            max_value=10, 
            value=5,
            step=1
        )
        
        current_mood = MOOD_EMOJI_MAP.get(mood_score, "🤔 Unknown")

    with col_mood_info:
        st.markdown(f"""
        <div style="background-color: #fff0f5; padding: 20px; border-radius: 12px; margin-top: 20px; text-align: center;">
            <p style="font-size: 1rem; color: #555; margin-bottom: 5px;">Your Current Mood Selection:</p>
            <h1 style="font-size: 3rem; color: #FF6F91; margin: 0;">{current_mood}</h1>
        </div>
        """, unsafe_allow_html=True)

    mood_note = st.text_area(
        "Optional: Add a brief note about why you feel this way (e.g., 'Had a good conversation with a friend', 'Stressed about homework').",
        placeholder="Enter your note here...",
        key="mood_note"
    )

    if st.button("Log My Mood", use_container_width=True):
        user_id = st.session_state.get("user_id")
        
        if save_mood_db(user_id, mood_score, mood_note):
            
            # Update local state immediately
            new_mood = {
                "date": datetime.now().isoformat(),
                "mood": mood_score,
                "note": mood_note
            }
            st.session_state["mood_history"].insert(0, new_mood)
            
            st.success(f"Mood Logged: {current_mood}!")
            
            # Simple AI suggestion based on mood
            if mood_score < 4:
                ai_suggestion = safe_generate(f"User logged a low mood score of {mood_score}. Give a 1-sentence supportive message and a small suggestion (e.g., breathing exercise).")
                st.warning(f"**A Note from Your Buddy:** {ai_suggestion}")
            
            # Clear note and rerun to update history
            st.session_state["mood_note"] = ""
            st.rerun()
        else:
            st.error("Failed to save mood. Check database connection.")

    st.markdown("---")
    st.subheader("Your Recent Mood History")
    
    if st.session_state["mood_history"]:
        df_mood = pd.DataFrame(st.session_state["mood_history"])
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.strftime('%Y-%m-%d %H:%M')
        df_mood['Mood'] = df_mood['mood'].apply(lambda x: MOOD_EMOJI_MAP.get(x, "N/A"))
        df_mood.rename(columns={'note': 'Note', 'date': 'Time Logged'}, inplace=True)
        
        st.dataframe(df_mood[['Time Logged', 'Mood', 'Note']].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Log your mood to see your history here.")

# -------------------- Wellness Check-in (PHQ-9) Page --------------------
def wellness_checkin_page():
    # Logic preserved
    st.title("🧠 Wellness Check-in")
    st.subheader("The PHQ-9 is a brief self-assessment tool. Your privacy is protected.")
    st.markdown("---")
    
    # Display last check-in date
    last_phq9_date = st.session_state.get("last_phq9_date")
    if last_phq9_date:
        st.info(f"Your last check-in was on: **{last_phq9_date}**.")
    
    # Use a form to capture all answers at once
    with st.form("phq9_form"):
        st.markdown("**Over the last two weeks, how often have you been bothered by the following problems?**")
        
        phq9_answers = {}
        for i, question in enumerate(PHQ9_QUESTIONS):
            # Use the question itself as the label
            answer = st.radio(
                f"**{question}**",
                options=list(PHQ9_SCORES.keys()),
                key=f"phq9_q_{i}",
                horizontal=True
            )
            phq9_answers[i] = PHQ9_SCORES[answer]
            
        submitted = st.form_submit_button("Submit Wellness Check-in", use_container_width=True)

    if submitted:
        total_score = sum(phq9_answers.values())
        
        # Interpretation logic (standard PHQ-9 cutoffs)
        if total_score >= 20:
            interpretation = "Severe Depression. Please seek professional help immediately. Contact a crisis line or trusted adult."
            st.error(f"**Score: {total_score}/27** - {interpretation} 🚨")
        elif total_score >= 15:
            interpretation = "Moderately Severe Depression. Professional support is strongly recommended."
            st.warning(f"**Score: {total_score}/27** - {interpretation} ⚠️")
        elif total_score >= 10:
            interpretation = "Moderate Depression. Consider talking to a counselor or therapist."
            st.info(f"**Score: {total_score}/27** - {interpretation} 💡")
        elif total_score >= 5:
            interpretation = "Mild Depression. Keep monitoring your mood and focus on positive self-care activities."
            st.success(f"**Score: {total_score}/27** - {interpretation} 🌱")
        else:
            interpretation = "Minimal Depression. You are doing well! Continue healthy habits."
            st.success(f"**Score: {total_score}/27** - {interpretation} 🎉")

        # Save to DB and update session state
        user_id = st.session_state.get("user_id")
        if save_phq9_db(user_id, total_score, interpretation):
            st.session_state["phq9_score"] = total_score
            st.session_state["phq9_interpretation"] = interpretation
            st.session_state["last_phq9_date"] = datetime.now().strftime("%Y-%m-%d")
            st.toast("Check-in saved!", icon="✅")
            st.rerun()
        else:
            st.error("Failed to save check-in. Note the score manually.")

# -------------------- AI Chat Page --------------------
def ai_chat_page():
    # Logic preserved
    st.title("🤖 AI Chat")
    st.subheader("Your supportive AI buddy is here to listen and offer non-judgemental advice.")
    
    # Check AI availability
    if not st.session_state.get("_ai_available"):
        st.error("The AI chat is currently disabled because the `OPENROUTER_API_KEY` is missing.")
        st.markdown("---")
        st.stop()

    # Chat display container (reversed order for new messages at bottom)
    chat_container = st.container(height=400, border=True)

    # Display chat messages in the container
    with chat_container:
        messages = st.session_state.chat_messages
        # Skip the system instruction at index 0
        display_messages = messages[1:] if messages and messages[0].get("role") == "system" else messages
        
        for message in display_messages:
            if message["role"] != "system":
                # Use a custom avatar for the assistant
                avatar = "🤖" if message["role"] == "assistant" else "👤"
                st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

    # Chat input at the bottom
    prompt = st.chat_input("Ask your buddy anything...")
    
    if prompt:
        # 1. Display user message immediately
        with chat_container:
            st.chat_message("user", avatar="👤").markdown(prompt)

        # 2. Add user message to state
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 3. Generate and stream AI response
        with st.spinner("Buddy is thinking..."):
            ai_response = safe_generate(prompt)

        # 4. Display AI response
        with chat_container:
            st.chat_message("assistant", avatar="🤖").markdown(ai_response)
        
        # 5. Add AI response to state and rerun to update chat display persistence
        st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
        st.rerun()

# -------------------- CBT Thought Record Page --------------------
def cbt_thought_record_page():
    # Logic preserved
    st.title("💡 CBT Thought Record")
    st.subheader("Challenging negative thoughts step-by-step.")
    st.markdown("---")

    st.markdown("Fill out the steps below to identify and reframe an automatic negative thought.")
    
    # Form for the CBT record
    with st.form("cbt_record_form"):
        
        # Step 1-3 (Input for the negative thought)
        st.markdown("### Part 1: Capturing the Thought")
        for i in range(3):
            st.session_state["cbt_thought_record"][i] = st.text_input(
                CBT_PROMPTS[i], 
                value=st.session_state["cbt_thought_record"][i],
                key=f"cbt_q_{i}"
            )
            
        st.markdown("---")
        st.markdown("### Part 2: Challenging the Thought")

        # Step 4 (Evidence FOR)
        st.session_state["cbt_thought_record"][3] = st.text_area(
            CBT_PROMPTS[3],
            value=st.session_state["cbt_thought_record"][3],
            key=f"cbt_q_{3}",
            height=100
        )
        
        # Step 5 (AI-Generated Evidence AGAINST will be displayed after submission)
        # Note: Input for this step is skipped as it's AI-generated
        
        st.markdown("---")
        st.markdown("### Part 3: The Balanced Conclusion")

        # Step 6 (Balanced Reframe)
        st.session_state["cbt_thought_record"][5] = st.text_area(
            CBT_PROMPTS[5],
            value=st.session_state["cbt_thought_record"][5],
            key=f"cbt_q_{5}",
            height=100,
            placeholder="e.g., 'Even though I struggled with one part, I still succeeded at others, and that one mistake doesn't define my worth.'"
        )

        submitted = st.form_submit_button("Complete Thought Record and Get Reframe", use_container_width=True)

    if submitted:
        if save_cbt_record(st.session_state["cbt_thought_record"]):
            st.rerun() # Rerun to display the newly saved card immediately

    # Display the most recently completed Thought Record 
    if st.session_state.get("last_reframing_card"):
        card = st.session_state["last_reframing_card"]
        st.markdown("## Your Reframing Card 🧠")
        
        with st.container(border=True):
            st.markdown(f"**Date:** {pd.to_datetime(card['date']).strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Emotion:** {card['emotion']}")
            st.markdown(f"**Automatic Negative Thought:** *{card['thought']}*")
            st.divider()
            
            st.markdown("### The Verdict:")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.info("#### Evidence FOR the Thought")
                st.markdown(card['evidence_for'])
            with col_b:
                # Use a specific AI-related element
                st.success("#### 🤖 Evidence AGAINST (AI Reframing)")
                st.markdown(card['ai_reframing'])
                
            st.divider()
            st.success("### ✅ Your Balanced Reframe")
            st.markdown(card['balanced_reframe'])
            
    st.markdown("---")
    st.subheader("Thought Record History")
    
    # Display the history of CBT records
    if st.session_state.get("cbt_history"):
        for record in st.session_state["cbt_history"][:5]: # Show top 5
             with st.expander(f"**{pd.to_datetime(record['date']).strftime('%Y-%m-%d')}** | Thought: {record['thought'][:50]}...", expanded=False):
                 st.markdown(f"**Situation:** {record['situation']}")
                 st.markdown(f"**Emotion:** {record['emotion']}")
                 st.markdown(f"**Balanced Reframe:** {record['balanced_reframe']}")
    else:
        st.info("Complete a Cognitive Thought Record to see your history.")

# -------------------- Mindful Breathing Page (UPDATED WITH VISUALS) --------------------
def mindful_breathing_page():
    st.title("🌬️ Mindful Breathing")
    st.subheader("Follow the circle: Grow to Inhale, Pause to Hold, Shrink to Exhale.")
    st.markdown("---")
    
    # Breathing parameters
    inhale_time = 4
    hold_time = 7
    exhale_time = 8
    
    breathing_state = st.session_state.get("breathing_state", "stop")
    
    def start_breathing():
        st.session_state["breathing_state"] = "running"
        # Only increment the goal counter when starting the session
        st.session_state["daily_goals"]["breathing_session"]["count"] = st.session_state["daily_goals"]["breathing_session"].get("count", 0) + 1
        st.rerun()

    def stop_breathing():
        st.session_state["breathing_state"] = "stop"
        st.rerun()

    if breathing_state == "stop":
        st.info("Click 'Start Session' to begin the 4-7-8 guided breathing exercise.")
        # Placeholder for the static circle display
        st.markdown(f"""
        <div class="breathing-circle" style="transform: scale(0.6); background-color: #FF9CC2;">
            CLICK START
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Session", use_container_width=True):
            start_breathing()

    if breathing_state == "running":
        
        # Placeholder for the dynamic circle
        circle_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if st.button("Stop Session", on_click=stop_breathing, use_container_width=True):
             # The on_click handler will trigger the stop_breathing function and rerender
             return # Exit the function early

        # --- The Breathing Loop ---
        try:
            for cycle in range(1, 4): # Run 3 full cycles for a good session
                
                # 1. Inhale (4s) - Circle grows
                with circle_placeholder:
                    st.markdown(f"""
                    <div class="breathing-circle inhale">
                        <p style="font-size: 2.5rem; margin: 0;">INHALE</p>
                        <p style="font-size: 1rem; margin: 0;">4 SECONDS</p>
                    </div>
                    """, unsafe_allow_html=True)
                with status_placeholder:
                    st.info(f"Cycle {cycle}/3: Breathe in deep...")
                time.sleep(inhale_time)
                
                # 2. Hold (7s) - Circle is large and steady
                with circle_placeholder:
                    st.markdown(f"""
                    <div class="breathing-circle hold">
                        <p style="font-size: 2.5rem; margin: 0;">HOLD</p>
                        <p style="font-size: 1rem; margin: 0;">7 SECONDS</p>
                    </div>
                    """, unsafe_allow_html=True)
                with status_placeholder:
                    st.warning(f"Cycle {cycle}/3: Hold your breath...")
                time.sleep(hold_time)

                # 3. Exhale (8s) - Circle shrinks
                with circle_placeholder:
                    st.markdown(f"""
                    <div class="breathing-circle exhale">
                        <p style="font-size: 2.5rem; margin: 0;">EXHALE</p>
                        <p style="font-size: 1rem; margin: 0;">8 SECONDS</p>
                    </div>
                    """, unsafe_allow_html=True)
                with status_placeholder:
                    st.success(f"Cycle {cycle}/3: Slowly release...")
                time.sleep(exhale_time)

            # End of session
            circle_placeholder.empty()
            status_placeholder.empty()
            st.success("Breathing Session Complete! Goal achieved. Take a moment to check in with how you feel.")
            stop_breathing() # Reset state
            
        except st.runtime.scriptrunner.StopException:
             # This is expected behavior when st.rerun() is called inside the loop
             pass
        except Exception:
             # Catch other exceptions
             circle_placeholder.empty()
             status_placeholder.empty()
             st.error("An error occurred during the breathing session.")
             stop_breathing()

    st.markdown("---")
    st.markdown("### The 4-7-8 Technique")
    st.markdown("""
    * **Inhale** quietly through your nose for **4 seconds** (Circle grows).
    * **Hold** your breath for a count of **7 seconds** (Circle stays large).
    * **Exhale** completely through your mouth for **8 seconds** (Circle shrinks).
    """)


# -------------------- IoT Dashboard (ECE) Page --------------------
def iot_dashboard_page():
    # Logic preserved
    st.title("❤️ IoT Dashboard (ECE Demo)")
    st.subheader("Simulating real-time Heart Rate (PPG) and Stress (GSR) data using Kalman filtering.")
    st.markdown("---")
    
    # Main columns for layout
    col_a, col_b = st.columns([2, 1])
    
    with col_b:
        st.subheader("Real-time Metrics")
        latest_data = st.session_state["latest_ece_data"]
        
        # Display latest HR
        hr = latest_data.get("filtered_hr", 75.0)
        hr_color = "#FF6F91"
        dashboard_metric("Filtered Heart Rate", f"{hr:.0f}", "BPM", icon="❤️", color=hr_color)

        # Display latest GSR/Stress
        gsr = latest_data.get("gsr_stress_level", 1.0)
        gsr_color = "#FF9CC2" if gsr < 1.5 else "#FFC107" if gsr < 2.5 else "#FF6F91"
        dashboard_metric("Stress Level (GSR)", f"{gsr:.2f}", "/5.0", icon="⚡", color=gsr_color)

        # Control button
        if st.session_state["ece_running"]:
            if st.button("Stop Simulation", key="stop_ece", use_container_width=True):
                st.session_state["ece_running"] = False
                st.rerun()
        else:
            if st.button("Start Simulation", key="start_ece", use_container_width=True):
                st.session_state["ece_running"] = True
                st.rerun()

    with col_a:
        st.subheader("Physiological Data Stream")
        
        # Placeholder for the chart and the table (if needed)
        chart_placeholder = st.empty()
        
        # Simulation logic
        if st.session_state["ece_running"]:
            
            # Use a time source (just milliseconds since epoch)
            current_time_ms = int(time.time() * 1000)
            
            # 1. Generate new noisy data
            raw_data = generate_simulated_physiological_data(current_time_ms)
            
            # 2. Apply Kalman Filter to the raw PPG signal (Heart Rate)
            kalman_hr, new_kalman_state = kalman_filter_simple(raw_data["raw_ppg_signal"], st.session_state["kalman_state"])
            
            # 3. Update session state
            st.session_state["kalman_state"] = new_kalman_state
            
            new_row = raw_data.copy()
            new_row["kalman_hr"] = kalman_hr
            
            # Update the latest data metric display
            st.session_state["latest_ece_data"] = {
                "filtered_hr": kalman_hr, 
                "gsr_stress_level": raw_data["gsr_stress_level"]
            }

            # Append new data to the DataFrame
            df = st.session_state["physiological_data"]
            # Ensure the row is a DataFrame for clean concatenation
            new_df_row = pd.DataFrame([new_row])
            # Limit the DataFrame size to the last 150 points for performance
            st.session_state["physiological_data"] = pd.concat([df, new_df_row], ignore_index=True).tail(150)
            df_plot = st.session_state["physiological_data"].copy()

            # Create the Plotly chart (HR)
            fig = px.line(
                df_plot, 
                x='time_ms', 
                y=['raw_ppg_signal', 'filtered_hr', 'kalman_hr'], 
                height=300, 
                title="Heart Rate Signal (Raw vs. Filtered)",
                labels={
                    "value": "Heart Rate (BPM)", 
                    "variable": "Signal Type", 
                    "time_ms": "Time (ms)"
                },
                color_discrete_map={
                    'raw_ppg_signal': 'rgba(255, 107, 107, 0.3)',  # Light Red (Noisy)
                    'filtered_hr': 'rgb(66, 133, 244)',           # Blue (Clean)
                    'kalman_hr': 'rgb(156, 39, 176)'              # Purple (Kalman)
                },
                template="plotly_white"
            )

            # Improve trace appearance
            fig.update_traces(
                line=dict(width=1), 
                selector=dict(name='raw_ppg_signal')
            )
            fig.update_traces(
                line=dict(width=3), 
                selector=dict(name='filtered_hr')
            )
            fig.update_traces(
                line=dict(width=2, dash='dot'), 
                selector=dict(name='kalman_hr')
            )
            
            # Display chart
            with chart_placeholder:
                st.plotly_chart(fig, use_container_width=True)

            # Control the update speed
            time.sleep(0.1) 
            
            # Crucial: Rerun to fetch the next data point
            st.rerun() 
            
        else:
            st.info("Simulation is stopped. Click 'Start Simulation' to see real-time data.")
            with chart_placeholder:
                # Display an empty or static plot when stopped
                st.dataframe(pd.DataFrame(columns=['Time', 'HR (Filtered)', 'Stress (GSR)']), use_container_width=True)

# -------------------- Report & Summary Page --------------------
def report_summary_page():
    # Logic preserved
    st.title("📊 Personal Wellness Report")
    st.subheader("Your insights over time, based on your logs.")
    st.markdown("---")

    mood_data = st.session_state["mood_history"]
    journal_data = st.session_state["daily_journal"]
    phq9_score = st.session_state.get("phq9_score")
    phq9_interpretation = st.session_state.get("phq9_interpretation")

    if not mood_data and not journal_data:
        st.warning("No data yet! Log your mood and journal to see your report.")
        return

    # --- Section 1: Mood Summary ---
    st.header("1. Mood and Emotional Trend")
    if mood_data:
        df_mood = pd.DataFrame(mood_data)
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.date
        df_mood.rename(columns={'mood': 'Mood Score'}, inplace=True)
        
        avg_mood = df_mood['Mood Score'].mean()
        latest_mood = df_mood['Mood Score'].iloc[0]
        
        st.markdown(f"**Average Mood Score (Overall):** **{avg_mood:.2f}/10**")
        st.markdown(f"**Latest Mood Score:** **{latest_mood}/10** ({MOOD_EMOJI_MAP.get(latest_mood)})")
        
        # Calculate weekly average (requires grouping by week, simplified to last 7 days mean)
        df_mood_7d = df_mood.head(7)
        if not df_mood_7d.empty:
             avg_mood_7d = df_mood_7d['Mood Score'].mean()
             st.markdown(f"**Average Mood Score (Last 7 Logs):** **{avg_mood_7d:.2f}/10**")


        # Visualization: Mood Score distribution
        mood_counts = df_mood['Mood Score'].value_counts().sort_index()
        fig_mood = px.bar(
            x=mood_counts.index, 
            y=mood_counts.values, 
            labels={'x': 'Mood Score', 'y': 'Count'}, 
            title='Frequency of Mood Scores Logged',
            color=mood_counts.index,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_mood, use_container_width=True)
    else:
        st.info("No mood logs available.")

    st.markdown("---")
    
    # --- Section 2: Journal & Sentiment Analysis ---
    st.header("2. Journaling and Reflection")
    if journal_data:
        df_journal = pd.DataFrame(journal_data)
        df_journal['date'] = pd.to_datetime(df_journal['date']).dt.date
        df_journal.rename(columns={'sentiment': 'Sentiment Score'}, inplace=True)

        avg_sentiment = df_journal['Sentiment Score'].mean()
        
        st.markdown(f"**Total Entries:** **{len(df_journal)}**")
        st.markdown(f"**Average Sentiment Score (Overall):** **{avg_sentiment:.2f}** (Range: -1.0 to 1.0)")

        # Visualization: Sentiment over time
        df_sentiment_plot = df_journal.sort_values('date').tail(30).reset_index(drop=True)
        fig_sentiment = px.line(
            df_sentiment_plot, 
            x='date', 
            y='Sentiment Score', 
            markers=True, 
            title='Sentiment Score of Journal Entries Over Time',
            template="plotly_white"
        )
        fig_sentiment.add_hline(y=0.0, line_dash="dot", annotation_text="Neutral", annotation_position="bottom right")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("No journal entries available.")

    st.markdown("---")
    
    # --- Section 3: Wellness Check-in (PHQ-9) ---
    st.header("3. Latest Wellness Check-in")
    if phq9_score is not None:
        st.metric(
            label="Latest PHQ-9 Score", 
            value=f"{phq9_score}/27", 
            delta_color="off"
        )
        st.info(f"**Interpretation:** {phq9_interpretation}")
        st.markdown(f"*(Based on check-in from {st.session_state.get('last_phq9_date', 'N/A')})*")
    else:
        st.info("Complete the **Wellness Check-in** page to see your latest score and interpretation.")
        
        
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
        elif current_page == "Report & Summary": 
            report_summary_page()
        elif current_page == "IoT Dashboard (ECE)": 
            iot_dashboard_page()
        else:
            st.warning("Page not found or not yet implemented.")