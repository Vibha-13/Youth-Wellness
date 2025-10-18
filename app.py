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

# ---------- Streamlit page config and LAYOUT SETUP (Modernized) ----------
st.set_page_config(
    page_title="HarmonySphere", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_page_and_layout():
    # --- CSS STYLING FOR SOFT PASTEL WELLNESS VIBE ---
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

/* 2. Streamlit TextArea */
textarea {
    color: #1E1E1E !important;
    -webkit-text-fill-color: #1E1E1E !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border: 2px solid #FFD6E0 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.3s ease-in-out;
}
textarea:focus {
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

/* 4. Custom Sidebar Colors/Style */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #fff0f5, #e0f7fa);
    box-shadow: 2px 0 10px rgba(0,0,0,0.05);
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

/* 6. Sidebar Status Tags */
.sidebar-status {
    padding: 6px 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
}
.status-connected { background-color: #D4EDDA; color: #155724; border-left: 4px solid #28A745; }
.status-local { background-color: #FFF3CD; color: #856404; border-left: 4px solid #FFC107; }

/* 7. Hide Streamlit Footer */
footer {
    visibility: hidden;
}

/* 8. Breathing Circle (Animated Calm Effect) */
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
    transition: all 4s ease-in-out;
}
.breathe-inhale {
    animation: scaleIn 4s infinite alternate ease-in-out;
}
.breathe-exhale {
    animation: scaleOut 6s infinite alternate ease-in-out;
}
@keyframes scaleIn {
    from { transform: scale(1); }
    to { transform: scale(2); }
}
@keyframes scaleOut {
    from { transform: scale(2); }
    to { transform: scale(1); }
}
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
        # FIX: The column name is 'id' in your schema, not 'user_id'.
        admin_client.table("profiles").insert({
            "id": new_user_id, # <-- CORRECTED FROM "user_id"
            "created_at": current_time
        }).execute()
        
        # If both inserts succeed, the function returns the ID
        return new_user_id
            
    except Exception as e:
        # st.error(f"DB Error: {e}") # Uncomment this line temporarily to see the real error if it still fails
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

# --- Placeholder functions for page content ---

def unauthenticated_home():
    st.title("Welcome to HarmonySphere 🧠")
    st.subheader(random.choice(QUOTES))
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <h3 style="color: #FF9CC2; margin-top: 0;">What is HarmonySphere?</h3>
            <p>HarmonySphere is your private wellness companion, designed to help you track your emotions, practice mindful techniques, and reframe negative thoughts using tools inspired by Cognitive Behavioral Therapy (CBT) and real-time biometric feedback (ECE). 🌱</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #FF9CC2;">
            <h4>🔐 Private & Secure</h4>
            <p>Your journal entries and personal data are kept confidential and are protected by robust database security (RLS).</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #FF9CC2;">
            <h4>🤖 AI-Powered Support</h4>
            <p>Engage in supportive chat or receive AI reframing suggestions for difficult thoughts, focused on safety and empathy.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style="padding: 20px; border-radius: 12px; background-color: #fff0f5; text-align: center; margin-top: 30px;">
            <h3 style="color: #5D54A4; margin-top: 0;">Access Your Dashboard</h3>
            <p>Please use the login form on the left sidebar to access the app's features.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.info("Remember: HarmonySphere is a support tool, not a substitute for medical advice.")

def homepage_panel():
    st.title("🏠 HarmonySphere Dashboard")
    st.subheader(f"Welcome back, {st.session_state['user_email'].split('@')[0].capitalize()}!")
    st.markdown("---")
    
    # --- PLANT HEALTH & QUOTE ---
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.subheader("Your Wellness Ecosystem 🌱")
        # Ensure plant health is calculated/set
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
            completed = goal["count"] >= goal["target"]
            card_style = "metric-card" + (" done" if completed else " pending")
            emoji = "✅" if completed else "⏳"
            
            st.markdown(f"""
            <div class="{card_style}" style="border-left: 5px solid {'#28A745' if completed else '#FFC107'};">
                <p style='font-size: 0.85rem; color: #555; margin-bottom: 5px;'>Goal: {goal['frequency']}</p>
                <h4 style="margin-top: 0; margin-bottom: 5px;">{goal['name']} {emoji}</h4>
                <p style="font-size: 0.9rem;">{goal['count']} / {goal['target']} Completed</p>
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

# Placeholder for other page functions (assuming they are defined elsewhere in the original code, but only providing the corrected main logic here)
def mindful_journaling_page():
    st.title("📝 Mindful Journaling")
    st.write("This is the Mindful Journaling page.")

def mood_tracker_page():
    st.title("📈 Mood Tracker")
    st.write("This is the Mood Tracker page.")

def wellness_checkin_page():
    st.title("🩺 Wellness Check-in")
    st.write("This is the Wellness Check-in page.")

def ai_chat_page():
    st.title("💬 AI Chat")
    st.write("This is the AI Chat page.")

def wellness_ecosystem_page():
    st.title("🌱 Wellness Ecosystem")
    st.write("This is the Wellness Ecosystem page.")

def mindful_breathing_page():
    st.title("🧘‍♀️ Mindful Breathing")
    st.write("This is the Mindful Breathing page.")

def cbt_thought_record_page():
    st.title("✍️ CBT Thought Record")
    st.write("This is the CBT Thought Record page.")

def journal_analysis_page():
    st.title("📊 Journal Analysis")
    st.write("This is the Journal Analysis page.")

def iot_dashboard_page():
    st.title("⚙️ IoT Dashboard (ECE)")
    st.write("This is the IoT Dashboard page.")

def report_summary_page():
    st.title("📄 Report & Summary")
    st.write("This is the Report & Summary page.")


# ---------- Sidebar Navigation and Auth (Modified) ----------
def sidebar_auth():
    st.sidebar.markdown("---")
    st.sidebar.header("Account")
    
    # --- Status Tags ---
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
    
    if st.session_state.get("logged_in"):
        st.sidebar.success(f"Logged in as: {st.session_state['user_email']}")
        if st.sidebar.button("Logout", key="logout_button", use_container_width=True):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                if not key.startswith("_"): # Don't clear resource caches
                    del st.session_state[key]
            st.session_state["logged_in"] = False
            st.session_state["page"] = "Home"
            st.rerun()
        return

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
                # IMPORTANT: We assume the RLS on 'users' table is correctly set to 'SELECT authenticated' or similar
                user_list = get_user_by_email_db(email) 
                if user_list:
                    user = user_list[0]

            if user or db_connected is False:
                # --- AUTHENTICATION SUCCESS ---
                st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                st.session_state["user_email"] = email
                st.session_state["logged_in"] = True

                if user and db_connected:
                    # Load ALL data from DB
                    with st.spinner("Loading your personalized wellness data..."):
                        # Ensure load_all_user_data is called with the RLS-secured client
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

                # --- SMOOTH TRANSITION START (NEW) ---
                st.sidebar.empty()
                st.sidebar.info("Preparing your dashboard. Please wait...")
                time.sleep(1.5) # Intentional delay for smooth transition feel
                st.rerun()
                # --- SMOOTH TRANSITION END ---

            else:
                # --- Registration Attempt ---
                if db_connected:
                    # FIX IS HERE: register_user_db now uses the correct 'id' column name
                    uid = register_user_db(email) 
                    
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.session_state["daily_goals"] = DEFAULT_GOALS.copy()
                        st.sidebar.success("New user registered and logged in! 🎉")
                        st.rerun() # Rerun to load authenticated state
                    else:
                        st.sidebar.error("Failed to register user in DB. Check secrets and Admin Key permissions.")
                else:
                    st.sidebar.error("User not found and DB is not connected. Cannot register.")
        else:
            st.sidebar.error("Please enter a valid email address.")


# Run auth section
sidebar_auth()

# ---------- MAIN APPLICATION LOGIC ----------
if not st.session_state.get("logged_in"):
    # --- UNAUTHENTICATED PAGES ---
    unauthenticated_home()

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
    else:
        st.warning("Page not found or not yet implemented.")