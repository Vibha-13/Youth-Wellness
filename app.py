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
}}

/* Conditional CSS to HIDE the sidebar immediately upon load when NOT logged in */
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
</style>
""", unsafe_allow_html=True)

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
        state = initialize_kalman() 
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
# This client uses the Service Role Key to bypass RLS for user creation
@st.cache_resource(show_spinner=False)
def get_supabase_admin_client():
    try:
        # Load the dedicated Service Role Key from secrets
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        key = st.secrets.get("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY"))
        
        if not url or not key:
            # If the Service Key is missing, registration will default to local or fail gracefully
            return None
        
        # Create client with service_role privileges (BYPASSES RLS)
        return create_client(url, key)
    except Exception as e:
        # st.error(f"Failed to initialize Admin Client: {e}") # Debug line
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

# !!! FIX APPLIED HERE: Only inserts into 'profiles' and uses the 'id' column. !!!
def register_user_db(email: str):
    """
    Inserts a new user entry into the 'profiles' table 
    using the dedicated Admin Client to bypass RLS.
    """
    admin_client = get_supabase_admin_client()
    
    if not admin_client:
        return None 
        
    new_user_id = str(uuid.uuid4())
    current_time = datetime.now().isoformat() 
    
    try:
        # NOTE: Removed the insert into the custom 'users' table to fix registration errors.

        # 1. Insert ONLY into the 'profiles' table 
        admin_client.table("profiles").insert({
            "id": new_user_id, # FIX: Uses the correct 'id' column from your schema
            "created_at": current_time
        }).execute()
        
        return new_user_id
            
    except Exception as e:
        # st.error(f"Registration failed on profiles insert: {e}") # Debugging hook
        return None

def get_user_by_email_db(email: str):
    """Searches the custom 'users' or 'profiles' table for an existing user."""
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return []
    try:
        # Try finding in the 'profiles' table
        res = supabase_client.table("profiles").select("id").eq("email", email).execute()
        if res.data:
             return res.data or []
        
        # Fallback to check a custom 'users' table if it exists (for compatibility)
        res = supabase_client.table("users").select("id").eq("email", email).execute()
        return res.data or []

    except Exception:
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

@st.cache_data(show_spinner=False)
def load_all_user_data(user_id, supabase_client):
    # Logic preserved
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
                for key in ["user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state", "daily_journal", "mood_history", "physiological_data", "ece_history", "plant_health", "cbt_history", "last_reframing_card"]:
                    if key in st.session_state:
                        # Skip if the key is already set to None or an empty container type, but ensure sensitive ones are cleared.
                        if key in ["user_id", "user_email", "phq9_score", "phq9_interpretation"]:
                            st.session_state[key] = None
                        elif key in ["kalman_state"]:
                            st.session_state[key] = initialize_kalman()
                        elif key in ["daily_journal", "mood_history", "ece_history", "cbt_history"]:
                            st.session_state[key] = []
                        elif key in ["physiological_data"]:
                            st.session_state[key] = pd.DataFrame(columns=["time_ms", "raw_ppg_signal", "filtered_hr", "gsr_stress_level"])
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
                    # --- AUTHENTICATION SUCCESS ---
                    st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True

                    # --- DATA LOADING (Transition Start) ---
                    if user and db_connected:
                        with st.spinner("Loading your personalized wellness data..."):
                            user_data = load_all_user_data(st.session_state["user_id"], st.session_state.get("_supabase_client_obj"))
                            
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
                    # --- 2. Registration Attempt ---
                    if db_connected:
                        uid = register_user_db(email) # Calls the fixed registration function
                        
                        if uid:
                            st.session_state["user_id"] = uid
                            st.session_state["user_email"] = email
                            st.session_state["logged_in"] = True
                            st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
                            st.success("New user registered and logged in! Redirecting...")
                            time.sleep(1.0)
                            st.session_state["page"] = "Home"
                            st.rerun()
                        else:
                            st.error("Failed to register user in DB. Check secrets or Service Key permissions.")
                    else:
                        st.error("User not found and DB is not connected. Cannot register.")
            else:
                st.error("Please enter a valid email address.")


def homepage_panel():
    st.title("🏠 HarmonySphere Dashboard")
    st.subheader(f"Welcome back, {st.session_state['user_email'].split('@')[0].capitalize()}!")
    st.markdown("---")
    
    # --- PLANT HEALTH & QUOTE ---
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.subheader("Your Wellness Ecosystem 🌱")
        calculate_plant_health()
        
        health = st.session_state["plant_health"]
        if health >= 85:
            plant_status = "Flourishing! 🌻"
        elif health >= 60:
            plant_status = "Healthy and steady. ✨"
        elif health >= 30:
            plant_status = "Needs a little attention. 💧"
        else:
            plant_status = "Wilting. Urgent care needed! 🚨"
            
        st.markdown(f"**Status:** *{plant_status}*")
        st.progress(health / 100, text=f"Health: {health:.1f}%")

    with col_b:
        st.markdown(f"#### Today's Inspiration")
        st.markdown(f"> *{random.choice(QUOTES)}*")
        
    st.markdown("---")
    
    # --- DAILY GOALS & STREAK ---
    st.subheader("Daily Wellness Goals")
    
    goal_cols = st.columns(3)
    
    for i, (key, goal) in enumerate(st.session_state["daily_goals"].items()):
        with goal_cols[i % 3]:
            # Use safe check just in case
            if not isinstance(goal, dict):
                continue
                
            completed = goal.get("count", 0) >= goal.get("target", 1)
            card_style = "metric-card" + (" done" if completed else " pending")
            emoji = "✅" if completed else "⏳"
            
            st.markdown(f"""
            <div class="{card_style}" style="border-left: 5px solid {'#28A745' if completed else '#FFC107'};">
                <p style='font-size: 0.85rem; color: #555; margin-bottom: 5px;'>Goal: {goal['frequency']}</p>
                <h4 style="margin-top: 0; margin-bottom: 5px;">{goal['name']} {emoji}</h4>
                <p style="font-size: 0.9rem;">{goal.get('count', 0)} / {goal.get('target', 1)} Completed</p>
            </div>
            """, unsafe_allow_html=True)

    # --- JOURNALING QUICK VIEW ---
    st.markdown("---")
    st.subheader("Journal Summary")
    if st.session_state["daily_journal"]:
        latest_entry = st.session_state["daily_journal"][0]
        sentiment_score = latest_entry["sentiment"]
        
        if sentiment_score >= 0.05:
            sentiment_text = "Positive 😊"
            color = "#28A745"
        elif sentiment_score <= -0.05:
            sentiment_text = "Negative 😞"
            color = "#DC3545"
        else:
            sentiment_text = "Neutral 😐"
            color = "#FFC107"

        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {color};">
            <p style='font-size: 0.85rem; color: #555; margin-bottom: 5px;'>Latest Entry: {pd.to_datetime(latest_entry['date']).strftime('%Y-%m-%d')}</p>
            <h4 style="margin-top: 0; margin-bottom: 5px;">Sentiment: {sentiment_text}</h4>
            <p style="font-size: 0.9rem;">"{latest_entry['text'][:150].strip()}..."</p>
            <a href="?page=Mindful Journaling" target="_self" style="color: #FF9CC2;">Continue Journaling &rarr;</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("You haven't made a journal entry yet. Head to **Mindful Journaling** to start!")
        
    # --- ECE DATA (Quick Glance) ---
    st.markdown("---")
    st.subheader("Real-Time Biofeedback (ECE)")
    latest_data = st.session_state["latest_ece_data"]
    
    ece_col1, ece_col2, ece_col3 = st.columns(3)
    
    with ece_col1:
        hr_color = "#007BFF" if 60 <= latest_data["filtered_hr"] <= 100 else "#DC3545"
        st.metric("Heart Rate (BPM)", f"{latest_data['filtered_hr']:.1f}", delta=None, delta_color="off")
    
    with ece_col2:
        stress_emoji = "😌" if latest_data["gsr_stress_level"] < 2.5 else ("🤨" if latest_data["gsr_stress_level"] < 5.0 else "🥵")
        st.metric("GSR Stress Level", f"{latest_data['gsr_stress_level']:.2f}", delta=None, delta_color="off")
        
    with ece_col3:
        if st.button("Start Biofeedback Session", key="start_ece_home_button", use_container_width=True):
            st.session_state["ece_running"] = True
            st.session_state["page"] = "IoT Dashboard (ECE)"
            st.rerun()

# --- FUNCTIONAL FEATURE: Mindful Journaling ---
def mindful_journaling_page():
    st.title("📝 Mindful Journaling")
    st.subheader("What's on your mind today?")
    st.caption("Writing down your thoughts and feelings can help you process your emotions.")

    user_id = st.session_state.get("user_id")
    
    # Check if a journal entry has already been completed today
    today = datetime.now().strftime("%Y-%m-%d")
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

# --- FUNCTIONAL FEATURE: Mood Tracker ---
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
        df_mood_daily = df_mood_daily.head(30).sort_values('date') # Show last 30 days

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
        fig.update_layout(yaxis_range=[1, 10.5])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log your first mood to see your trend chart!")

# --- FUNCTIONAL FEATURE: CBT Thought Record ---
def cbt_thought_record_page():
    st.title("✍️ CBT Thought Record")
    st.subheader("Challenge Negative Thoughts")
    st.caption("This tool, based on Cognitive Behavioral Therapy, helps you identify and reframe unhelpful thought patterns.")

    # --- CBT Form ---
    with st.form(key="cbt_form", clear_on_submit=True):
        st.markdown("### 🔑 Thought Record Steps")
        
        # Use text_area for better input experience
        for i, prompt in enumerate(CBT_PROMPTS):
            # Check if the text is already stored in the session state dictionary
            default_value = st.session_state["cbt_thought_record"].get(i, "")
            st.session_state["cbt_thought_record"][i] = st.text_area(
                prompt,
                key=f"cbt_step_{i}",
                height=100 if i not in [3, 4] else 150,
                value=default_value
            )

        st.markdown("---")
        cbt_submitted = st.form_submit_button("Analyze & Reframe Thought", use_container_width=True)
        
    if cbt_submitted:
        # Pass the current state to the saving function
        cbt_data = st.session_state["cbt_thought_record"].copy()
        
        if cbt_data.get(0, "") and cbt_data.get(2, ""): # Ensure Situation and Thought are filled
            save_cbt_record(cbt_data) # This function handles AI generation and local state update
            # The save_cbt_record function calls st.success() and sets st.session_state["last_reframing_card"]
        else:
            st.error("Please fill out the **Situation** and the **Negative Thought** to proceed.")

    # --- Display Last Reframing Card ---
    if st.session_state["last_reframing_card"]:
        st.markdown("---")
        st.subheader("Your Last Reframed Thought ✨")
        
        last_record = st.session_state["last_reframing_card"]
        
        # Display the core record inputs
        st.info(f"**Situation:** {last_record['situation']} | **Emotion:** {last_record['emotion']}")
        st.warning(f"**Negative Thought:** {last_record['thought']}")

        # Display AI-Generated Counter-Evidence
        st.markdown("#### The AI's Counter-Evidence (Evidence AGAINST)")
        st.markdown(last_record['ai_reframing'])
        
        # Display User's Final Reframed Thought
        st.success(f"**Balanced Reframe (Your Takeaway):** {last_record['balanced_reframe']}")
        
    st.markdown("---")
    
    # --- History (Future implementation) ---
    if st.session_state["cbt_history"]:
        st.caption(f"You have {len(st.session_state['cbt_history'])} thought records saved.")
    else:
        st.caption("Complete a thought record to see your history here.")

# --- FUNCTIONAL FEATURE: Mindful Breathing ---
def mindful_breathing_page():
    st.title("🧘‍♀️ Mindful Breathing")
    st.subheader("Follow the breath cycle to calm your nervous system.")
    st.caption("A 60-second session will complete your daily goal.")

    # CSS for the Breathing Animation
    st.markdown("""
    <style>
    /* Keyframes for the expanding and contracting breath circle */
    @keyframes breath {
        0% { transform: scale(1); background-color: #FFD6E0; }
        40% { transform: scale(1.5); background-color: #FF9CC2; }
        60% { transform: scale(1.5); background-color: #FF9CC2; }
        100% { transform: scale(1); background-color: #FFD6E0; }
    }

    .breathing-circle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 400px; /* Gives space for the animation */
        margin-top: 20px;
        position: relative;
    }

    .breathing-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(255, 156, 194, 0.7);
        transition: all 0.5s ease-out;
        will-change: transform, background-color;
        /* Animation is applied dynamically via Python */
    }

    .instruction-text {
        position: absolute;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E1E1E;
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    .instruction-text.show {
        opacity: 1;
    }

    </style>
    """, unsafe_allow_html=True)

    # --- Session State Management ---
    if "breathing_start_time" not in st.session_state:
        st.session_state["breathing_start_time"] = None
    if "breathing_duration" not in st.session_state:
        st.session_state["breathing_duration"] = 60 # seconds for goal

    # --- Button Logic ---
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.session_state["breathing_state"] == "stop":
            if st.button("Start Breathing", use_container_width=True, key="start_breath"):
                st.session_state["breathing_state"] = "running"
                st.session_state["breathing_start_time"] = time.time()
                st.rerun()
        else:
            if st.button("Stop Session", use_container_width=True, key="stop_breath"):
                st.session_state["breathing_state"] = "stop"
                st.session_state["breathing_start_time"] = None
                st.rerun()
    
    # --- Animation and Tracking ---
    placeholder = st.empty()
    
    if st.session_state["breathing_state"] == "running":
        
        # Calculate time elapsed and remaining
        elapsed_time = time.time() - st.session_state["breathing_start_time"]
        remaining_time = max(0, st.session_state["breathing_duration"] - elapsed_time)
        
        # Determine current phase (4s inhale, 6s exhale, 10s cycle)
        cycle_time = elapsed_time % 10
        if 0 <= cycle_time < 4:
            instruction = "INHALE (4s)"
            animation = "breath_in"
        elif 4 <= cycle_time < 10:
            instruction = "EXHALE (6s)"
            animation = "breath_out"
        else:
            instruction = "HOLD"
            animation = "hold"
        
        # Inject animation and instruction text
        with placeholder.container():
            st.markdown(f"""
            <div class="breathing-circle-container">
                <div class="breathing-circle" style="animation: breath 10s infinite ease-in-out;"></div>
                <div class="instruction-text show" id="instruction_text">{instruction}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Time Remaining: **{int(remaining_time)} seconds**")
            
        # Check for session completion
        if remaining_time <= 0:
            st.session_state["breathing_state"] = "completed"
            st.session_state["daily_goals"]["breathing_session"]["count"] = 1
            st.success("🎉 Breathing Session Complete! Goal achieved for today.")
            # Clear the animation by rerunning
            time.sleep(1) # Visual pause
            st.rerun()
            
        # Re-run every second to update timer and phase
        time.sleep(1)
        st.rerun()

    elif st.session_state["breathing_state"] == "completed":
        st.balloons()
        st.markdown("<div class='breathing-circle-container'><p>Goal Complete!</p></div>", unsafe_allow_html=True)
        st.session_state["breathing_state"] = "stop" # Reset for next session

    else:
        # Initial 'stop' state
        if st.session_state["daily_goals"]["breathing_session"]["count"] >= st.session_state["daily_goals"]["breathing_session"]["target"]:
             st.success("✅ Daily Breathing Goal Completed!")
        else:
             st.info("Start a 60-second breathing session to complete your daily goal.")
             
        st.markdown(f"""
        <div class="breathing-circle-container">
            <div class="breathing-circle" style="transform: scale(1); background-color: #FFD6E0;"></div>
            <div class="instruction-text show" id="instruction_text">Ready to Begin</div>
        </div>
        """, unsafe_allow_html=True)

# --- FUNCTIONAL FEATURE: AI Chat ---
def ai_chat_page():
    st.title("💬 AI Chat: Your Wellness Buddy")
    st.caption("Chat with an empathetic AI designed to listen and provide non-judgemental support.")

    # 1. Display Chat Messages
    # Start loop from 1 to skip the system message (index 0)
    # The initial 'Hello' message is at index 1
    for message in st.session_state["chat_messages"][1:]:
        avatar = "🧠" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # 2. Handle User Input
    if prompt := st.chat_input("Ask me anything or tell me how you're feeling..."):
        
        # Add user message to state and display immediately
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Buddy is thinking..."):
                # Call the safe_generate function with the user's prompt
                ai_response = safe_generate(prompt, max_tokens=600)
                
                # Check for critical safety response and apply styling
                if ai_response.startswith("**🛑 STOP."):
                    st.error(ai_response)
                else:
                    st.markdown(ai_response)

        # Add AI response to state
        st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response})


# Placeholder functions for remaining features
def wellness_checkin_page():
    st.title("🩺 Wellness Check-in")
    st.write("### PHQ-9 Check-in Logic Goes Here.")
    st.info("This is where the user answers the PHQ-9 questions and the score is interpreted.")

def wellness_ecosystem_page():
    st.title("🌱 Wellness Ecosystem")
    st.write("### Plant Gamification/Interaction Logic Goes Here.")
    st.info("Show the user's plant based on `st.session_state['plant_health']` and allow them to interact.")

def journal_analysis_page():
    st.title("📊 Journal Analysis")
    st.write("### Data Visualization Logic Goes Here.")
    st.info("Plot mood scores vs. journal sentiment scores to show correlation.")

def iot_dashboard_page():
    st.title("⚙️ IoT Dashboard (ECE)")
    st.write("### Real-time ECE Data Plotting Goes Here.")
    st.info("Simulate or display ECE data, showing Kalman filtered HR and GSR stress levels.")

def report_summary_page():
    st.title("📄 Report & Summary")
    st.write("### Summary Report Generation Logic Goes Here.")
    st.info("A comprehensive PDF/Downloadable report of the user's weekly/monthly data.")


# ---------- Sidebar Navigation and Auth (Icon-Based) ----------
def sidebar_auth():
    st.sidebar.markdown("---")
    st.sidebar.header("System Status")
    
    # --- Status Tags (Kept from original code) ---
    ai_status_class = "status-connected" if st.session_state.get("_ai_available") else "status-local"
    db_status_class = "status-connected" if st.session_state.get("_db_connected") else "status-local"
    st.sidebar.markdown(
        f"<div class='sidebar-status {ai_status_class}'>AI: <b>{'CONNECTED' if st.session_state.get('_ai_available') else 'LOCAL'}</b></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"<div class='sidebar-status {db_status_class}'>DB: <b>{'CONNECTED' if st.session_state.get('_db_connected') else 'NOT CONNECTED'}</b></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    if st.session_state.get("logged_in"):
        st.sidebar.caption(f"Welcome, {st.session_state['user_email'].split('@')[0].capitalize()}!")
        
        # Logged-in Navigation with Icons
        st.sidebar.header("Features")
        
        # Define Pages with Aesthetic Emojis/Icons
        pages = {
            "🏠 Home": "Home",
            "📝 Mindful Journaling": "Mindful Journaling",
            "📈 Mood Tracker": "Mood Tracker",
            "✍️ CBT Thought Record": "CBT Thought Record",
            "💬 AI Chat": "AI Chat",
            "🧘‍♀️ Mindful Breathing": "Mindful Breathing",
            "🌱 Wellness Ecosystem": "Wellness Ecosystem",
            "⚙️ IoT Dashboard (ECE)": "IoT Dashboard (ECE)",
        }
        
        # Use a radio button for navigation (cleaner UX than buttons)
        page_keys = list(pages.keys())
        current_index = page_keys.index(next((k for k, v in pages.items() if v == st.session_state["page"]), "🏠 Home"))
        
        selected_page_key = st.sidebar.radio(
            "Go to:", 
            options=page_keys, 
            index=current_index,
            key="navigation_radio"
        )
        st.session_state["page"] = pages[selected_page_key]
        
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", key="logout_button", use_container_width=True):
            # Clear all session state variables for a clean logout
            for key in list(st.session_state.keys()):
                if not key.startswith("_"): 
                    # Only clear keys that are not system/cached resources
                    if key in st.session_state and key not in ["logged_in", "show_splash", "page"]:
                        del st.session_state[key]
                        
            st.session_state["logged_in"] = False
            st.session_state["show_splash"] = True # Go back to splash
            st.session_state["page"] = "Home"
            st.rerun()

    else:
        # Unauthenticated: Sidebar only shows status
        st.sidebar.info("Log in on the main screen to start.")

# Run auth and navigation section (this must run first)
sidebar_auth()

# Create a main placeholder for the application content
app_placeholder = st.empty()

# ---------- MAIN APPLICATION LOGIC (Triple Flow) ----------
with app_placeholder.container():
    
    if st.session_state.get("show_splash"):
        # 1. Show Splash Screen first
        app_splash_screen()
        
    elif not st.session_state.get("logged_in"):
        # 2. Transition to Centered Login
        unauthenticated_home()

    else:
        # 3. Transition to Authenticated Dashboard
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
        else:
            st.warning("Page not found or not yet implemented.")