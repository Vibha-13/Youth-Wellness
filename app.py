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

# ---------- Streamlit page config and LAYOUT SETUP (Modernized) ----------
st.set_page_config(
    page_title="HarmonySphere", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_page_and_layout():
    # --- CSS STYLING FOR MODERN LOOK & TRANSITIONS ---
    st.markdown("""
    <style>
    /* 1. Global Background and Typography */
    .stApp { 
        background: #f7f9fb; /* Very light gray/off-white background */
        color: #1E1E1E; 
        font-family: 'Poppins', sans-serif; 
    }
    .main .block-container { 
        padding: 1.5rem 4rem; /* Adjusted padding */
    }
    
    /* 2. CRITICAL: Target the Streamlit Text Area's internal input element */
    textarea {
        color: black !important; 
        -webkit-text-fill-color: black !important; 
        opacity: 1 !important; 
        background-color: white !important; 
        border: 1px solid #ccc !important; 
        border-radius: 8px !important;
    }

    /* 3. Custom Card Style (The Core Mobile App Look) */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        /* Soft, subtle shadow for depth */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); 
        background-color: #ffffff; /* White card background */
        transition: transform 0.2s, box-shadow 0.2s; /* Hover transition */
        margin-bottom: 20px;
        border: none;
    }
    .metric-card:hover { 
        transform: translateY(-3px); /* Lifts the card slightly */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12); /* Deeper shadow on hover */
        cursor: pointer;
    }

    /* 4. Custom Sidebar Colors/Style */
    [data-testid="stSidebar"] {
        background-color: #ffffff; /* White sidebar */
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
    }
    
    /* 5. Primary Button Style (Vibrant and Rounded) */
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
    
    /* 6. Custom Sidebar Status */
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
    
    /* 7. Hiding Streamlit Footer */
    footer {
        visibility: hidden;
    }

    /* 8. Breathing Circle Styles (Used existing one for consistency) */
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

    </style>
    """,
    unsafe_allow_html=True,
)

# Call the setup function early in the main script flow
setup_page_and_layout()


# ---------- ECE HELPER FUNCTIONS (KALMAN FILTER) ----------
# ... (rest of your ECE functions remain here) ...


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
# ... (rest of your state setup remains here) ...

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
# ... (rest of your AI functions remain here) ...

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

        # 4. Also insert into 'profiles' table (FIX: Removed the non-existent "username" column)
        admin_client.table("profiles").insert({
            "id": new_user_id,
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

# ... (rest of your DB save functions remain here) ...

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
    # ... (rest of your load function remains here) ...
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
    # ... (rest of your plant health function remains here) ...
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
    # ... (rest of your goal reset function remains here) ...
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


# ---------- Sidebar Navigation and Auth (Modified) ----------
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
                    # --- AUTHENTICATION SUCCESS ---
                    st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    
                    if user and db_connected: 
                        # Load ALL data from DB
                        with st.spinner("Loading your personalized wellness data..."):
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
                        uid = register_user_db(email)
                        if uid:
                            st.session_state["user_id"] = uid
                            st.session_state["user_email"] = email
                            st.session_state["logged_in"] = True
                            st.sidebar.success("Registered & logged in. 🎉")
                            
                            # --- SMOOTH TRANSITION START (NEW) ---
                            st.sidebar.empty()
                            st.sidebar.info("Setting up your profile. Please wait...")
                            time.sleep(1.5) # Intentional delay
                            st.rerun() 
                            # --- SMOOTH TRANSITION END ---
                            
                        else:
                            st.sidebar.error("Registration failed. Try a different email or check DB connection.")
                    else:
                        st.sidebar.error("Could not find user and DB not connected to register.")
            else:
                st.sidebar.warning("Please enter a valid email address.")
    
    else:
        # Logout logic
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout", key="sidebar_logout_btn", use_container_width=True):
            # Reset major state variables
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation", "kalman_state", "daily_journal", "mood_history", "physiological_data", "ece_history", "daily_goals", "plant_health", "chat_messages", "cbt_history", "last_reframing_card"]:
                if key in st.session_state:
                    del st.session_state[key]
            
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

# ---------- SPLASH SCREEN LOGIC (NEW) ----------
def show_splash_screen():
    # Clear all content placeholders before showing splash
    st.empty() 

    # Center the content vertically and horizontally
    col = st.columns([1, 4, 1])
    with col[1]:
        st.markdown(
            f"""
            <div style="height: 50vh; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <h1 style="color: #5D54A4; font-size: 3.5rem;">HarmonySphere 🧘</h1>
                <p style="font-size: 1.2rem; color: #6c757d;">Loading wellness resources and connecting to the cloud...</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Use the built-in spinner to show activity
        with st.spinner('Preparing your session...'):
            time.sleep(2) # Show the splash screen for 2 seconds

# Check if the app has loaded before
if "app_loaded" not in st.session_state:
    show_splash_screen()
    st.session_state["app_loaded"] = True
    st.rerun() # Rerun to load the actual content after the splash screen is done
# ---------- END SPLASH SCREEN LOGIC ----------


# ---------- FEATURE PAGE FUNCTIONS (Built and working) ---------- 

# --- 1. Homepage (MODERNIZED CARD STRUCTURE) --- 
def homepage_panel():
    
    # --- CRITICAL SAFETY FIX: Ensure session state lists are not None ---
    # This prevents the TypeError you encountered if the session state was deleted/set to None during a rerun.
    if st.session_state.get("daily_journal") is None:
        st.session_state["daily_journal"] = []
    if st.session_state.get("mood_history") is None:
        st.session_state["mood_history"] = []
    # -------------------------------------------------------------------
    
    st.markdown(f"<h1>Your Wellness Sanctuary <span style='color: #5D54A4;'>🧠</span></h1>", unsafe_allow_html=True)
    st.caption("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")

    # --- CRISIS ALERT ---
    if st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] >= PHQ9_CRISIS_THRESHOLD:
        st.error("🚨 **CRISIS ALERT:** Your last Wellness Check-in indicated a high level of distress. Please prioritize contacting a helpline or trusted adult immediately. Your safety is paramount.")
    
    st.markdown("---")

    # --- Fetching Dummy/Real Data for Cards (Now Safe from None) ---
    total_entries = len(st.session_state["daily_journal"]) # No need for .get(..., []) since we checked above
    df_mood = pd.DataFrame(st.session_state["mood_history"])
    avg_mood_7d = df_mood.head(7)['mood'].mean() if not df_mood.empty else 6.0 
    
    # Calculate a simple streak (for visualization)
    streak = 0
    if st.session_state.get("mood_history"):
        dates = pd.to_datetime([item['date'] for item in st.session_state['mood_history']]).date
        today = datetime.now().date()
        
        # Check today first
        if today in dates:
            streak = 1
            current_date = today - timedelta(days=1)
            while current_date in dates:
                streak += 1
                current_date -= timedelta(days=1)
        
    current_streak = streak

    # --- METRIC CARDS ---
    st.subheader("Quick Glance")
    col1, col2, col3 = st.columns(3)

    # --- Card 1: Total Journal Entries ---
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5em; font-weight: 700; color: #5D54A4;">{total_entries}</div>
            <div style="font-size: 1.0em; color: #6c757d;">Total Journal Entries</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Card 2: Average Sentiment Score ---
    with col2:
        sentiment_color = "#4CAF50" if avg_mood_7d >= 6.5 else "#FFC107" if avg_mood_7d >= 5.5 else "#FF5722"
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
            <div style="font-size: 2.5em; font-weight: 700; color: #2196F3;">{current_streak} Days 🔥</div>
            <div style="font-size: 1.0em; color: #6c757d;">Current Streak</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Your Daily Focus ✨")
    # ... (Your existing code for personalized AI insight and goals nudges) ...
    # Calculate relevant metrics
    if st.session_state.get("mood_history"):
        df_mood = pd.DataFrame(st.session_state["mood_history"])
    else:
        df_mood = pd.DataFrame()
        
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
            elif avg_mood_7d is not None and avg_mood_7d <= 5.0:
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
        # Convert to datetime and calculate 7-day rolling average
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        df_mood = df_mood.sort_values(by='date').reset_index(drop=True)
        df_mood['7-Day Avg'] = df_mood['mood'].rolling(window=7, min_periods=1).mean()
        
        # Plotly chart in a clean card
        fig = px.line(
            df_mood, 
            x="date", 
            y="7-Day Avg", 
            title="Mood Trend (7-Day Rolling Average)",
            labels={"date": "Date", "7-Day Avg": "Average Mood Score (1-11)"},
            template="plotly_white"
        )
        fig.update_traces(line_color='#5D54A4', line_width=3)
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f7f9fb",
            font=dict(family="Poppins")
        )

        with st.container():
             st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mood data yet. Log your first mood to see your trends!")
# --- 2. Mindful Journaling ---
def mindful_journaling_page():
    st.markdown(f"<h1>Mindful Journaling <span style='color: #5D54A4;'>📝</span></h1>", unsafe_allow_html=True)
    st.caption("A space to process your thoughts. Your entries are private and analyzed for insights.")
    
    # 1. Journal Input
    with st.form("journal_form", clear_on_submit=True):
        entry_text = st.text_area("What is on your mind today? (Be honest, this is for you)", height=300, key="journal_input")
        
        submitted = st.form_submit_button("Analyze & Save Entry", use_container_width=True)
        
        if submitted and entry_text:
            user_id = st.session_state["user_id"]
            
            with st.spinner('Analyzing sentiment and saving your entry...'):
                # Analyze Sentiment
                sentiment = sentiment_compound(entry_text)
                
                # Save to DB
                if st.session_state.get("_db_connected") and user_id:
                    success = save_journal_db(user_id, entry_text, sentiment)
                    if success:
                        st.success("Entry saved successfully!")
                    else:
                        st.warning("Entry saved locally. Could not connect to database for saving.")
                
                # Save to local state
                new_entry = {"date": datetime.now().isoformat(), "text": entry_text, "sentiment": sentiment}
                st.session_state["daily_journal"].insert(0, new_entry)
                
                # Update Goals
                st.session_state["daily_goals"]["journal_entry"]["count"] = 1
                check_and_reset_goals()

            # 2. Display Analysis
            st.subheader("🤖 AI Insight: Your Mood Snapshot")
            
            with st.container(border=True):
                col_score, col_feedback = st.columns([1, 3])
                
                with col_score:
                    # Determine color based on sentiment score
                    if sentiment >= 0.3:
                        color = "#4CAF50" # Green
                        mood_text = "Positive"
                    elif sentiment <= -0.3:
                        color = "#FF5722" # Red
                        mood_text = "Negative"
                    else:
                        color = "#FFC107" # Yellow/Orange
                        mood_text = "Neutral"
                        
                    st.markdown(f"""
                        <div style="text-align: center; padding: 10px; border-radius: 8px; background-color: {color}20;">
                            <div style="font-size: 2em; font-weight: 700; color: {color};">{sentiment:.3f}</div>
                            <div style="font-size: 0.9em; color: {color};">{mood_text} Score</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_feedback:
                    if sentiment >= 0.3:
                        st.markdown("🎉 **Feeling good!** Your entry reflects a **positive** tone. Remember to capture these moments and what led to them. Keep going!")
                    elif sentiment <= -0.3:
                        st.markdown("🫂 **I hear the heaviness.** Your entry suggests a **negative** emotional load. It's okay to feel this way. Consider doing a **Mindful Breathing** session or using the **CBT Thought Record** to unpack a difficult thought.")
                    else:
                        st.markdown("🤔 **A balanced entry.** The tone is mostly **neutral**. If you're holding back, remember this is a safe, judgment-free zone. Try a quick, simple thought reframing exercise.")

    st.subheader("Recent Entries")
    if st.session_state["daily_journal"]:
        for entry in st.session_state["daily_journal"][:5]:
            date_obj = pd.to_datetime(entry['date'])
            date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            sentiment = entry['sentiment']
            
            if sentiment >= 0.3:
                border_color = "#4CAF50"
            elif sentiment <= -0.3:
                border_color = "#FF5722"
            else:
                border_color = "#FFC107"

            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {border_color};">
                <p style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">{date_str} | Sentiment: {sentiment:.3f}</p>
                <p style="white-space: pre-wrap;">{entry['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Your journal is empty! Start writing your first entry above.")


# --- 3. Mood Tracker ---
def mood_tracker_page():
    st.markdown(f"<h1>Mood Tracker <span style='color: #5D54A4;'>📈</span></h1>", unsafe_allow_html=True)
    st.caption("Rate your mood on a scale of 1 to 10 and track your emotional shifts over time.")
    
    # 1. Mood Input Form
    with st.form("mood_form", clear_on_submit=True):
        st.subheader("How are you feeling right now?")
        
        # Use a slider from 1 to 10
        mood_score = st.slider(
            "Select your mood score (1=Agonizing, 10=Joyful)",
            min_value=1,
            max_value=10,
            value=6,
            step=1,
            format="%d - " + MOOD_EMOJI_MAP.get(10) # Placeholder text, actual emoji map is used in helper
        )
        
        # Display real-time emoji feedback
        st.markdown(f"**Selected Mood:** <span style='font-size: 1.5em;'>{MOOD_EMOJI_MAP.get(mood_score, '❓')}</span>", unsafe_allow_html=True)
        
        note = st.text_input("Quick note (optional, e.g., 'had a good chat with a friend')")
        
        submitted = st.form_submit_button("Log Mood", use_container_width=True)
        
        if submitted:
            user_id = st.session_state["user_id"]
            
            with st.spinner('Logging mood...'):
                if st.session_state.get("_db_connected") and user_id:
                    success = save_mood_db(user_id, mood_score, note)
                    if success:
                        st.success("Mood logged successfully!")
                    else:
                        st.warning("Mood logged locally. Could not connect to database for saving.")
                        
                # Save to local state
                new_mood = {"date": datetime.now().isoformat(), "mood": mood_score, "note": note}
                st.session_state["mood_history"].insert(0, new_mood)
                
                # Update Goals
                st.session_state["daily_goals"]["log_mood"]["count"] = 1
                check_and_reset_goals()
            
    st.subheader("Mood History and Trends")
    
    # 2. Mood Trend Chart
    df_mood = pd.DataFrame(st.session_state["mood_history"])
    
    if not df_mood.empty:
        # Prepare data for plotting
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.date
        df_mood = df_mood.sort_values(by='date', ascending=True).drop_duplicates(subset=['date'], keep='last')
        
        # Create a simple line plot
        fig = px.line(
            df_mood, 
            x="date", 
            y="mood", 
            title="Daily Mood Over Time",
            labels={"date": "Date", "mood": "Mood Score (1-10)"},
            template="plotly_white",
            markers=True
        )
        
        fig.update_layout(
            yaxis_range=[0.5, 10.5], 
            yaxis_tickvals=list(MOOD_EMOJI_MAP.keys())[:10],
            yaxis_ticktext=[MOOD_EMOJI_MAP[i] for i in list(MOOD_EMOJI_MAP.keys())[:10]],
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f7f9fb",
            font=dict(family="Poppins")
        )

        with st.container():
             st.plotly_chart(fig, use_container_width=True)
             
        # 3. Recent Mood Logs
        st.subheader("Recent Logs")
        for log in st.session_state["mood_history"][:5]:
            date_obj = pd.to_datetime(log['date'])
            date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            mood_score = log['mood']
            mood_emoji = MOOD_EMOJI_MAP.get(mood_score, '❓')
            
            # Simple color coding for card border
            if mood_score >= 7:
                border_color = "#4CAF50" # Happy (Green)
            elif mood_score <= 4:
                border_color = "#FF5722" # Sad (Red)
            else:
                border_color = "#FFC107" # Neutral (Yellow)
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {border_color};">
                <p style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">{date_str}</p>
                <p style="font-size: 1.2em;">**Mood:** {mood_emoji} ({mood_score}/10)</p>
                {f'<p style="white-space: pre-wrap;">**Note:** {log["note"]}</p>' if log['note'] else ''}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No mood logs yet. Use the form above to record your first mood!")


# --- 4. Wellness Check-in (PHQ-9) ---
def wellness_checkin_page():
    st.markdown(f"<h1>Wellness Check-in <span style='color: #5D54A4;'>🩺</span></h1>", unsafe_allow_html=True)
    st.caption("A quick and confidential check-in based on the PHQ-9. Recommended weekly.")
    
    # Display last check-in status
    if st.session_state["phq9_score"] is not None:
        st.info(f"**Last Check-in:** {st.session_state['last_phq9_date']} | **Score:** {st.session_state['phq9_score']} ({st.session_state['phq9_interpretation']} severity)")
        
    st.subheader("Patient Health Questionnaire (PHQ-9)")
    st.warning("Please answer the following questions based on how you have felt over the **last two weeks**.")

    with st.form("phq9_form"):
        phq9_answers = {}
        total_score = 0
        
        for i, question in enumerate(PHQ9_QUESTIONS):
            # Use the index as the key in the form, and a dedicated key for st.radio
            answer = st.radio(
                question, 
                list(PHQ9_SCORES.keys()), 
                key=f"phq9_q_{i}", 
                index=0
            )
            phq9_answers[i] = answer
            total_score += PHQ9_SCORES[answer]
            
        submitted = st.form_submit_button("Submit Check-in", use_container_width=True)
        
        if submitted:
            # 1. Calculate Interpretation
            interpretation = "Unknown"
            for (low, high), desc in PHQ9_INTERPRETATION.items():
                if low <= total_score <= high:
                    interpretation = desc
                    break
            
            st.session_state["phq9_score"] = total_score
            st.session_state["phq9_interpretation"] = interpretation
            st.session_state["last_phq9_date"] = datetime.now().strftime("%Y-%m-%d")
            
            # 2. Check for Crisis
            # Question 9 is about suicide ideation
            q9_index = SUICIDE_IDEATION_QUESTION_INDEX 
            q9_answer = phq9_answers[q9_index]
            q9_score = PHQ9_SCORES[q9_answer]

            st.subheader("Your Results")
            st.info(f"**Total Score:** {total_score} - **Severity:** {interpretation}")

            # 3. Crisis Redirection
            if q9_score > 0 or total_score >= PHQ9_CRISIS_THRESHOLD:
                st.error("🚨 **IMMEDIATE ATTENTION:** Based on your answers, please prioritize reaching out for professional support immediately. Your safety is paramount.")
                st.markdown("""
                ### **Immediate Crisis Resources:**
                * **Call/Text 988** (Suicide & Crisis Lifeline - US/Canada)
                * **Text HOME to 741741** (Crisis Text Line)
                * Go to your nearest Emergency Room or call a trusted adult.
                """)
            else:
                st.success("Your well-being is important. Remember to follow up with your progress.")
            
            # 4. Save to DB/Local
            user_id = st.session_state["user_id"]
            if st.session_state.get("_db_connected") and user_id:
                 with st.spinner("Saving results to the database..."):
                    save_phq9_db(user_id, total_score, interpretation)


# --- 5. AI Chat ---
def ai_chat_page():
    st.markdown(f"<h1>AI Wellness Buddy <span style='color: #5D54A4;'>💬</span></h1>", unsafe_allow_html=True)
    st.caption("A non-judgmental space to share your thoughts. The AI is trained to be empathetic and supportive.")
    
    if not st.session_state.get("_ai_available"):
        st.warning("⚠️ **AI Status:** The AI model is currently unavailable (API Key missing/invalid). Using a local, canned-response fallback.")

    # 1. Display Chat History
    for message in st.session_state["chat_messages"]:
        if message["role"] != "system":
            # Use Streamlit's built-in chat elements
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 2. Chat Input
    prompt = st.chat_input("Say something to your buddy...")
    
    if prompt:
        user_id = st.session_state["user_id"]
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                ai_response = safe_generate(prompt)
                st.markdown(ai_response)
                
            # Update chat history in session state
            st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response})
            
            # Save the full conversation piece to the journal history for later analysis (optional)
            journal_text = f"**AI Chat Session:** User: {prompt} | Buddy: {ai_response}"
            sentiment = sentiment_compound(prompt) # Analyze user's last sentiment only
            
            if st.session_state.get("_db_connected") and user_id:
                save_journal_db(user_id, journal_text, sentiment)
            else:
                st.session_state["daily_journal"].insert(0, {"date": datetime.now().isoformat(), "text": journal_text, "sentiment": sentiment})


# --- 6. Wellness Ecosystem (Gamification) ---
def wellness_ecosystem_page():
    st.markdown(f"<h1>Wellness Ecosystem <span style='color: #5D54A4;'>🌱</span></h1>", unsafe_allow_html=True)
    st.caption("Your daily self-care efforts help nurture your personal well-being plant.")

    st.subheader("Plant Health Score")
    health = st.session_state["plant_health"]
    
    # Determine emoji and color based on health
    if health >= 80:
        emoji = "🌳"
        color = "#4CAF50" # Dark Green
        status_text = "Thriving! Keep up the great work."
    elif health >= 50:
        emoji = "🌿"
        color = "#FFC107" # Yellow-Green
        status_text = "Stable. Needs consistent attention."
    else:
        emoji = "🥀"
        color = "#FF5722" # Red
        status_text = "Needs urgent care. Focus on the goals below."

    # Custom HTML for Plant Display
    st.markdown(f"""
    <div class="metric-card plant-container" style="background-color: {color}15; border: 1px solid {color}50;">
        <div class="plant-emoji" style="transform: scale({health/100 * 1.5 + 0.5});">{emoji}</div>
        <p style="font-size: 1.5em; font-weight: 700; color: {color}; margin-top: 10px;">{status_text}</p>
        <p style="font-size: 0.9em; color: #6c757d;">Health: {health:.1f}%</p>
        <div class="plant-health-bar">
            <div class="plant-health-fill" style="width: {health}%; background-color: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Daily Goals")
    st.caption(f"Goals reset daily. Last reset: {st.session_state['daily_goals']['log_mood']['last_reset']}")
    
    goals = st.session_state["daily_goals"]
    
    for key, goal in goals.items():
        is_complete = goal["count"] >= goal["target"]
        
        if is_complete:
            icon = "✅"
            style = "border-left: 5px solid #4CAF50; opacity: 0.7;"
            text_color = "#4CAF50"
        else:
            icon = "⏳"
            style = "border-left: 5px solid #FFC107;"
            text_color = "#333"

        st.markdown(f"""
        <div class="metric-card" style="{style}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1em; font-weight: 600; color: {text_color};">{icon} {goal['name']}</span>
                <span style="font-size: 1.1em; font-weight: 600; color: #5D54A4;">{goal['count']}/{goal['target']}</span>
            </div>
            <p style="font-size: 0.8em; color: #6c757d; margin-top: 5px;">Frequency: {goal['frequency']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Optional: Force a recalculation button
    if st.button("Recalculate Plant Health"):
        check_and_reset_goals()
        st.experimental_rerun()


# --- 7. Mindful Breathing ---
def mindful_breathing_page():
    st.markdown(f"<h1>Mindful Breathing <span style='color: #5D54A4;'>🧘‍♀️</span></h1>", unsafe_allow_html=True)
    st.caption("Follow the circle to regulate your breath. Use the 4-7-8 technique (Inhale 4, Hold 7, Exhale 8).")
    
    # Button to start/stop the session
    if st.session_state["breathing_state"] == "stop":
        if st.button("Start 3-Minute Session", use_container_width=True):
            st.session_state["breathing_state"] = "running"
            st.experimental_rerun()
    else:
        if st.button("Stop Session", use_container_width=True):
            st.session_state["breathing_state"] = "stop"
            
            # Update Goals on completion
            st.session_state["daily_goals"]["breathing_session"]["count"] = 1
            check_and_reset_goals()
            st.success("Mindful Breathing session complete! Goal achieved.")
            st.experimental_rerun()
            
    if st.session_state["breathing_state"] == "running":
        # Placeholder for the circle animation
        placeholder = st.empty()
        
        # --- Breathing Animation Loop (Needs custom HTML/JS to run fully smoothly) ---
        # Streamlit requires a hack to run persistent animation. We'll simulate with updates.
        
        # Custom HTML/CSS/JS for the animation using st.markdown
        # Note: This animation is simplified and relies on CSS animation, but the timing is hard to control perfectly in Streamlit.
        
        st.markdown("""
        <div style="text-align: center;">
            <div class="breathing-circle breathe-inhale" id="breathing-display">
                <span id="breathing-text">Inhale (4s)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Loop to update the text state
        
        text_placeholder = st.empty()
        # This will run for approx 3 minutes (6 cycles * 30 seconds total per cycle)
        total_cycles = 6
        seconds_per_cycle = 4 + 7 + 8 # Inhale 4, Hold 7, Exhale 8 = 19 seconds per full 4-7-8 cycle
        
        start_time = time.time()
        
        for cycle in range(1, total_cycles + 1):
            if st.session_state["breathing_state"] == "stop":
                break
            
            # Inhale (4 seconds)
            text_placeholder.info(f"Cycle {cycle}/{total_cycles}: **INHALE** slowly... (4 seconds)")
            time.sleep(4)
            if st.session_state["breathing_state"] == "stop": break

            # Hold (7 seconds)
            text_placeholder.warning(f"Cycle {cycle}/{total_cycles}: **HOLD** it gently... (7 seconds)")
            time.sleep(7)
            if st.session_state["breathing_state"] == "stop": break

            # Exhale (8 seconds)
            text_placeholder.success(f"Cycle {cycle}/{total_cycles}: **EXHALE** completely... (8 seconds)")
            time.sleep(8)
            if st.session_state["breathing_state"] == "stop": break
            
            # Pause/Transition (1 second)
            text_placeholder.empty()
            time.sleep(1) 
        
        if st.session_state["breathing_state"] == "running":
            # Auto-stop after completion
            st.session_state["breathing_state"] = "stop"
            st.experimental_rerun()
            

# --- 8. CBT Thought Record ---
def cbt_thought_record_page():
    st.markdown(f"<h1>CBT Thought Record <span style='color: #5D54A4;'>✍️</span></h1>", unsafe_allow_html=True)
    st.caption("Challenging negative thoughts using the Cognitive Behavioral Therapy (CBT) framework.")
    
    st.warning("Focus only on one specific negative thought and follow the steps below.")
    
    # 1. Thought Record Form
    with st.form("cbt_form", clear_on_submit=False):
        
        cbt_data_col = {}
        
        for i, prompt in enumerate(CBT_PROMPTS):
            # Use text_area for prompts 1, 3, 4, 5, 6. Use text_input for Emotion (2).
            if i in [1]: # Emotion (Step 2)
                cbt_data_col[i] = st.text_input(prompt, key=f"cbt_prompt_{i}", value=st.session_state["cbt_thought_record"].get(i, ""))
            elif i in [4]: # AI Reframing (Step 5 - This will be calculated by AI, skip user input)
                # Skip the AI step in the form, but keep the index placeholder
                continue
            else:
                cbt_data_col[i] = st.text_area(prompt, height=50 if i in [2] else 100, key=f"cbt_prompt_{i}", value=st.session_state["cbt_thought_record"].get(i, ""))
            
            # Update session state on change
            st.session_state["cbt_thought_record"][i] = cbt_data_col[i]

        submitted = st.form_submit_button("Get AI Reframing & Save Record", use_container_width=True)
        
        if submitted:
            # Re-read data from session state (updated by the widget callbacks)
            final_data = st.session_state["cbt_thought_record"]
            
            with st.spinner('Analyzing thought and generating counter-evidence...'):
                 save_cbt_record(final_data)
                 # Rerun to show the new card below and clear the form
                 st.experimental_rerun()
            

    # 2. Display Last Reframing Card
    if st.session_state["last_reframing_card"]:
        st.subheader("Last Completed Reframing Card")
        card = st.session_state["last_reframing_card"]
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #5D54A4; background-color: #F0F0FF;">
            <p style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">Date: {pd.to_datetime(card['date']).strftime('%Y-%m-%d %H:%M')}</p>
            
            <p><strong>1. Situation:</strong> {card['situation']}</p>
            <p><strong>3. Negative Thought:</strong> <span style="color: #FF5722; font-weight: 600;">{card['thought']}</span></p>
            <hr style="border-top: 1px solid #ccc; margin: 10px 0;">
            
            <p><strong>5. AI's Counter-Evidence (Evidence AGAINST the thought):</strong></p>
            <div style="background-color: #FFFFFF; padding: 10px; border-radius: 8px; border: 1px solid #ddd;">
            {card['ai_reframing']}
            </div>
            
            <p style="margin-top: 15px;"><strong>6. Your Balanced Reframe:</strong> <span style="color: #4CAF50; font-weight: 600;">{card['balanced_reframe']}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Record History")
    if len(st.session_state["cbt_history"]) > 1:
        for history_card in st.session_state["cbt_history"][1:5]:
            st.caption(f"Thought: {history_card['thought']} | Date: {pd.to_datetime(history_card['date']).strftime('%Y-%m-%d')}")
    elif len(st.session_state["cbt_history"]) == 1:
        st.caption("History will appear here after your second entry.")
    else:
        st.info("No completed thought records yet.")


# --- 9. Journal Analysis ---
def journal_analysis_page():
    st.markdown(f"<h1>Journal Analysis <span style='color: #5D54A4;'>📊</span></h1>", unsafe_allow_html=True)
    st.caption("In-depth visualization of your journal sentiment and topic trends.")
    
    if not st.session_state["daily_journal"]:
        st.info("No journal entries to analyze yet. Write something in **Mindful Journaling**!")
        return

    df_journal = pd.DataFrame(st.session_state["daily_journal"])
    df_journal['date'] = pd.to_datetime(df_journal['date'])
    df_journal = df_journal.sort_values(by='date', ascending=True)

    # --- 1. Sentiment Over Time ---
    st.subheader("Sentiment Trend Over Time")
    
    # Calculate rolling average for smoothing
    df_journal['Rolling_Sentiment'] = df_journal['sentiment'].rolling(window=3, min_periods=1).mean()
    
    fig_sentiment = px.line(
        df_journal, 
        x="date", 
        y="Rolling_Sentiment", 
        title="3-Entry Rolling Average Sentiment Score",
        labels={"date": "Date", "Rolling_Sentiment": "Sentiment Score"},
        template="plotly_white",
        markers=True
    )
    
    fig_sentiment.update_traces(line_color='#2196F3', line_width=2)
    fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral Line", annotation_position="bottom right")

    with st.container(border=True):
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # --- 2. Sentiment Distribution (Histogram) ---
    st.subheader("Sentiment Distribution")
    
    fig_hist = px.histogram(
        df_journal,
        x="sentiment",
        nbins=15,
        title="Frequency of Journal Sentiment Scores",
        labels={"sentiment": "Sentiment Score (VADER)"},
        template="plotly_white"
    )
    fig_hist.update_traces(marker_color='#5D54A4', opacity=0.8)
    fig_hist.update_layout(bargap=0.1)
    
    with st.container(border=True):
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- 3. Key Statistics ---
    st.subheader("Key Statistics")
    avg_sent = df_journal['sentiment'].mean()
    
    col_avg, col_pos, col_neg = st.columns(3)
    
    with col_avg:
        st.metric("Overall Avg. Sentiment", f"{avg_sent:.3f}")
    with col_pos:
        st.metric("Most Positive Entry Score", f"{df_journal['sentiment'].max():.3f}", delta="Top Entry")
    with col_neg:
        st.metric("Most Negative Entry Score", f"{df_journal['sentiment'].min():.3f}", delta="Bottom Entry", delta_color="inverse")
        
    st.markdown("---")
    
    # --- 4. Deepest Dive (Most Negative Entry) ---
    st.subheader("Deepest Dive: Most Negative Entry")
    min_sentiment_entry = df_journal.loc[df_journal['sentiment'].idxmin()]
    
    st.markdown(f"""
    <div class="metric-card" style="border: 1px solid #FF5722; background-color: #FF572210;">
        <p style="font-size: 0.9em; color: #FF5722; margin-bottom: 5px;">Date: {min_sentiment_entry['date'].strftime('%Y-%m-%d %H:%M')}</p>
        <p style="font-size: 1.5em; font-weight: 700; color: #FF5722;">Score: {min_sentiment_entry['sentiment']:.3f}</p>
        <p style="white-space: pre-wrap; margin-top: 10px; color: #333;">{min_sentiment_entry['text']}</p>
    </div>
    """, unsafe_allow_html=True)


# --- 10. IoT Dashboard (ECE) ---
def iot_dashboard_page():
    st.markdown(f"<h1>IoT Dashboard (ECE) <span style='color: #5D54A4;'>⚙️</span></h1>", unsafe_allow_html=True)
    st.caption("Simulated real-time physiological data (Heart Rate and Stress Level).")
    
    # Placeholder for the data chart and metrics
    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    if st.session_state["ece_running"]:
        if st.button("Stop Live Simulation", key="stop_ece_btn", use_container_width=True):
            st.session_state["ece_running"] = False
            st.experimental_rerun()
    else:
        if st.button("Start Live Simulation", key="start_ece_btn", use_container_width=True):
            st.session_state["ece_running"] = True
            st.experimental_rerun()

    # The Live Loop
    if st.session_state["ece_running"]:
        
        # 1. Generate new data point
        current_time_ms = time.time() * 1000
        new_data = generate_simulated_physiological_data(current_time_ms)
        
        # 2. Apply Kalman Filter to HR (PPG)
        kalman_state = st.session_state["kalman_state"]
        
        # Use the raw PPG signal as the noisy measurement (z_meas)
        filtered_hr, new_kalman_state = kalman_filter_simple(new_data["raw_ppg_signal"], kalman_state)
        
        # Update the data point with the filtered HR
        new_data["filtered_hr"] = filtered_hr
        st.session_state["kalman_state"] = new_kalman_state
        
        # 3. Update DataFrame
        df_new = pd.DataFrame([new_data])
        st.session_state["physiological_data"] = pd.concat([st.session_state["physiological_data"], df_new], ignore_index=True)
        
        # Keep only the last 100 points for performance
        if len(st.session_state["physiological_data"]) > 100:
            st.session_state["physiological_data"] = st.session_state["physiological_data"].iloc[-100:]
            
        # 4. Update latest ECE data
        st.session_state["latest_ece_data"] = {
            "filtered_hr": filtered_hr,
            "gsr_stress_level": new_data["gsr_stress_level"]
        }
        
        df_plot = st.session_state["physiological_data"].copy()
        df_plot["time_sec"] = df_plot["time_ms"].apply(lambda x: x / 1000)

        # 5. Update Metrics
        with metric_placeholder.container():
            st.subheader("Real-time Vitals")
            col_hr, col_stress = st.columns(2)
            
            latest_hr = filtered_hr
            latest_stress = new_data["gsr_stress_level"]
            
            # HR display
            hr_color = "#4CAF50" if latest_hr < 90 else "#FF5722"
            col_hr.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {hr_color};">
                <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">Filtered Heart Rate</p>
                <p style="font-size: 2em; font-weight: 700; color: {hr_color};">{latest_hr:.1f} <span style="font-size: 0.5em;">BPM</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Stress display
            stress_level_text = "Low" if latest_stress < 1.5 else "Moderate" if latest_stress < 2.5 else "High"
            stress_color = "#4CAF50" if latest_stress < 1.5 else "#FFC107" if latest_stress < 2.5 else "#FF5722"
            col_stress.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {stress_color};">
                <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 5px;">GSR Stress Level</p>
                <p style="font-size: 2em; font-weight: 700; color: {stress_color};">{latest_stress:.2f} <span style="font-size: 0.5em;">({stress_level_text})</span></p>
            </div>
            """, unsafe_allow_html=True)

        # 6. Update Chart
        with chart_placeholder.container():
            st.subheader("Physiological Signal Visualization")
            
            # Combine HR and Stress for dual-axis plot
            fig = px.line(
                df_plot, 
                x='time_sec', 
                y=['filtered_hr', 'gsr_stress_level'], 
                title='Heart Rate & Stress Over Time',
                labels={'time_sec': 'Time (s)', 'value': 'Value', 'variable': 'Signal'},
                template="plotly_white",
                height=450
            )
            
            # Use two axes
            fig.update_layout(
                yaxis=dict(title='Heart Rate (BPM)', title_font=dict(color="#4CAF50")),
                yaxis2=dict(title='Stress Level (GSR)', overlaying='y', side='right', title_font=dict(color="#2196F3")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            # Set colors
            fig.for_each_trace(lambda t: t.update(yaxis="y") if t.name == 'filtered_hr' else t.update(yaxis="y2"))
            fig.data[0].update(line=dict(color='#4CAF50', width=3), name='Heart Rate (Filtered)')
            fig.data[1].update(line=dict(color='#2196F3', width=3), name='Stress Level (GSR)')

            st.plotly_chart(fig, use_container_width=True)

        # 7. Rerun for next data point
        time.sleep(1) 
        st.rerun()


# --- 11. Report & Summary ---
def report_summary_page():
    st.markdown(f"<h1>Report & Summary <span style='color: #5D54A4;'>📄</span></h1>", unsafe_allow_html=True)
    st.caption("A consolidated view of your overall wellness and key actionable insights.")
    
    st.subheader("Overall Wellness Metrics")
    
    # 1. Prepare Data
    df_mood = pd.DataFrame(st.session_state.get("mood_history", []))
    df_journal = pd.DataFrame(st.session_state.get("daily_journal", []))
    
    # 2. Main Metrics
    total_days = (pd.to_datetime(df_mood['date']).dt.date.nunique() if not df_mood.empty else 0)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Total Active Days", f"{total_days} days")
    with col_b:
        st.metric("Total Journal Entries", len(df_journal))
    with col_c:
        st.metric("Last PHQ-9 Score", st.session_state.get("phq9_score") or "N/A", help=st.session_state.get("phq9_interpretation") or "")

    st.markdown("---")
    
    # 3. Actionable Insights
    st.subheader("Actionable Insights & Trends")
    
    # Journal Trend Insight
    if not df_journal.empty:
        avg_sent = df_journal['sentiment'].mean()
        if avg_sent > 0.3:
            st.success("✅ **Journal Trend:** Your journaling shows a consistently positive sentiment (Avg: {:.2f}). Continue focusing on gratitude and positive reframing.".format(avg_sent))
        elif avg_sent < -0.3:
            st.error("🛑 **Journal Trend:** Sentiment is trending low (Avg: {:.2f}). This indicates high emotional distress. It's highly recommended to utilize the **CBT Thought Record** or seek professional help.".format(avg_sent))
        else:
            st.info("💡 **Journal Trend:** Your sentiment is mostly neutral. Try to be more descriptive in your emotions to get deeper insights.")
            
    # Mood Insight
    if not df_mood.empty:
        avg_mood = df_mood['mood'].mean()
        if avg_mood < 5:
             st.warning("⚠️ **Mood Trend:** Your average mood ({:.1f}/10) is below target. Focus on achieving your daily goals to lift your health score.".format(avg_mood))
        else:
             st.success("🎉 **Mood Trend:** Excellent average mood ({:.1f}/10). This stability suggests your current self-care routine is working.".format(avg_mood))
             
    # PHQ-9 Risk Insight
    if st.session_state.get("phq9_score") is not None and st.session_state["phq9_score"] >= PHQ9_CRISIS_THRESHOLD:
        st.error(f"🚨 **HIGH RISK ALERT:** PHQ-9 score of {st.session_state['phq9_score']} requires immediate support.")
        
    st.markdown("---")
    
    # 4. Crisis Resource Callout (Persistent)
    with st.container(border=True):
        st.markdown("""
        ### Need Immediate Support?
        Remember, this app is NOT a substitute for professional help.
        * **Call/Text 988** (Suicide & Crisis Lifeline - US/Canada)
        * **Text HOME to 741741** (Crisis Text Line)
        """, unsafe_allow_html=True)


# --- FALLBACK/UNAUTH PAGE ---
def unauthenticated_home():
    st.markdown(f"<h1>Welcome to HarmonySphere <span style='color: #5D54A4;'>🧠</span></h1>", unsafe_allow_html=True)
    st.subheader("Your AI-Powered Youth Wellness Companion")
    st.caption("A private, judgment-free space for journaling, mood tracking, and CBT.")

    st.markdown("---")

    # Use a two-column layout for a clean landing page look
    col_feat, col_auth_prompt = st.columns([2, 1])

    with col_feat:
        st.markdown("""
        ### Key Features:
        * **AI Wellness Buddy:** A safe, empathetic AI that listens and offers supportive coping strategies.
        * **CBT Thought Records:** Guided exercises to challenge and reframe negative automatic thoughts.
        * **Wellness Ecosystem:** Gamified goals to keep you motivated and track your progress.
        * **Mindful Tools:** Breathing exercises, Mood Tracker, and in-depth Journal Analysis.
        """)

        # Placeholder for visual appeal (since we can't fetch images)
        st.markdown("""
        <div style='background-color: #d1d8e0; height: 300px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #333;'>
            [Placeholder: Image of wellness tools or abstract visual]
        </div>
        """, unsafe_allow_html=True)
        st.caption("Wellness is a journey.")

    with col_auth_prompt:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #5D54A4; background-color: #F0F0FF;">
            <h3 style="color: #5D54A4; margin-top: 0;">Access Your Dashboard</h3>
            <p>Please use the login form on the left sidebar to access the app's features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Remember: HarmonySphere is a support tool, not a substitute for medical advice.")


# ---------- MAIN APPLICATION LOGIC ----------
if not st.session_state.get("logged_in"):
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
        st.error(f"Page '{current_page}' not found.")