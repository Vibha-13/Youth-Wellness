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
    10: "🥰 Ecstatic"
}

# Wellness Check-in Mapping
WELLNESS_OPTIONS = {
    0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"
}


# ---------- CORE INITIALIZATION & UTILITIES ----------

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Initializes and returns the regular Supabase client."""
    try:
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY")) 
        
        if not url or not key:
            st.session_state["_db_connected"] = False
            return None
            
        st.session_state["_db_connected"] = True
        return create_client(url, key)
    except Exception as e:
        st.session_state["_db_connected"] = False
        print(f"ERROR initializing Supabase Client: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_supabase_admin_client():
    """
    Initializes a Supabase client using the Service Role Key (Admin Key).
    This client is used for secure user registration and reliable lookup, bypassing RLS.
    """
    try:
        # Load URL and SERVICE_KEY securely
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        # Use a distinct key for the Service Role Key (Admin Key)
        key = st.secrets.get("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY")) 
        
        if not url or not key:
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


@st.cache_resource(show_spinner=False)
def setup_ai_model():
    """Initializes the AI client using OpenRouter."""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
        
        if not api_key:
            st.session_state["_ai_connected"] = False
            return None

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        st.session_state["_ai_connected"] = True
        return client
        
    except Exception as e:
        st.session_state["_ai_connected"] = False
        print(f"ERROR initializing AI Client: {e}")
        return None

def initialize_session_state():
    """Sets up default session state variables."""
    # CRITICAL FIX: Clear cache to prevent UnhashableParamError from stale cache data
    try:
        load_all_user_data.clear()
    except AttributeError:
        # This can happen on first run before the function is fully defined
        pass
        
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if "_supabase_client_obj" not in st.session_state:
        st.session_state["_supabase_client_obj"] = get_supabase_client()
    if "_ai_client_obj" not in st.session_state:
        st.session_state["_ai_client_obj"] = setup_ai_model()
    if "_db_connected" not in st.session_state:
        st.session_state["_db_connected"] = False
    if "_ai_connected" not in st.session_state:
        st.session_state["_ai_connected"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "system", "content": "You are Harmony, a youth wellness coach focused on CBT and mindfulness. Your tone is supportive, encouraging, and non-judgmental. Keep responses concise and actionable. Do not give medical advice."}]
    if "phq9_score" not in st.session_state:
        st.session_state["phq9_score"] = 0
    if "latest_ece_data" not in st.session_state:
        st.session_state["latest_ece_data"] = {"hr": 70, "stress": 1.0}
    if "plant_health" not in st.session_state:
        st.session_state["plant_health"] = 75 # Initial Health (0-100)
    if "show_splash" not in st.session_state:
        # Show splash screen on first load
        st.session_state["show_splash"] = True 


def generate_simulated_physiological_data():
    """Generates a small batch of simulated HR and GSR data."""
    
    # 1. Base Heart Rate (influenced by mood)
    mood_score = st.session_state.get("latest_mood_score", 5)
    # Inverse correlation: Lower mood (1) -> Higher base HR (e.g., 85), Higher mood (10) -> Lower base HR (e.g., 65)
    base_hr = 75 - (mood_score - 5) * 2.5 
    hr_noise = random.gauss(0, 3) # Add some noise
    hr_value = base_hr + hr_noise
    
    # 2. GSR/Stress Simulation (correlated with base HR and overall phq9 score)
    phq9_score = st.session_state.get("phq9_score") or 0
    # Inverse correlation: Lower base HR and lower PHQ-9 lead to lower GSR/stress (closer to 1.0)
    gsr_base = 1.0 + (base_hr / 100.0) + 0.5 * (phq9_score / 27.0)
    gsr_noise = 0.5 * random.gauss(0, 1) # Add some noise to GSR
    # FIX APPLIED HERE: Changed 'gr_base' to 'gsr_base' (previous fix)
    gsr_value = gsr_base + gsr_noise 
    
    # Clip stress to reasonable limits
    stress_value = max(1.0, min(gsr_value, 5.0))
    
    # Add high-frequency noise for the raw PPG measurement
    ppg_raw = [0.5 * np.sin(2 * np.pi * hr_value / 60 * t) + random.uniform(-0.1, 0.1) for t in np.linspace(0, 2, 100)]
    
    return {
        "hr": round(hr_value, 1),
        "stress": round(stress_value, 2), # Simplified stress level
        "ppg_raw": ppg_raw
    }

# ---------- DATABASE HELPER FUNCTIONS ----------

# CRITICAL FIX: The get_user_by_email_db function now uses the Admin Client 
# to bypass RLS and ensures the login flow works for existing users.
def get_user_by_email_db(email: str):
    """
    Searches the database for an existing user's ID using their email.
    Uses the ADMIN CLIENT to bypass RLS, ensuring a reliable lookup.
    """
    # Use the RLS-bypassing Admin Client for reliable lookup
    supabase_client = get_supabase_admin_client()
    
    if not supabase_client:
        return []
        
    try:
        # Query the 'users' table (confirmed to hold the email constraint)
        res = supabase_client.table("users").select("id, email").eq("email", email).execute()
        
        return res.data or []

    except Exception as e:
        # st.error(f"CRITICAL ADMIN LOOKUP FAIL: {e}") # Uncomment for debugging
        return []


def register_user_db(email: str) -> str | None:
    """Inserts a new user entry into the 'users' and 'profiles' tables."""
    admin_client = get_supabase_admin_client()
    
    if not admin_client:
        st.error("Admin Client Initialization Error: Please ensure `SUPABASE_SERVICE_KEY` and `SUPABASE_URL` are set correctly in your Streamlit secrets file.")
        return None 
        
    new_user_id = str(uuid.uuid4())
    current_time = datetime.now().isoformat() 
    
    try:
        # 1. Insert into 'users' table 
        admin_client.table("users").insert({
            "id": new_user_id,
            "email": email,
            "created_at": current_time 
        }).execute()

        # 2. Also insert into 'profiles' table
        admin_client.table("profiles").insert({
            "id": new_user_id,
            "created_at": current_time,
            "email": email # Note: Assuming profiles table needs email for display
        }).execute()
        
        return new_user_id
            
    except Exception as e:
        # Detailed error is now printed to the user if it's a database-level failure
        st.error(f"DB Insert Error: {e}") 
        return None

def save_mood_db(user_id, mood_score: int, note: str) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("mood_logs").insert({
            "user_id": user_id, 
            "mood": mood_score, 
            "note": note, 
            "date": datetime.now().isoformat()
        }).execute()
        return True
    except Exception:
        return False

def save_journal_db(user_id, entry_text: str, sentiment: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("journal_entries").insert({
            "user_id": user_id, 
            "entry_text": entry_text, 
            "sentiment_score": sentiment,
            "date": datetime.now().isoformat()
        }).execute()
        return True
    except Exception:
        return False

def save_wellness_checkin_db(user_id, phq9_score: int, answers: dict) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("wellness_checkins").insert({
            "user_id": user_id, 
            "phq9_score": phq9_score, 
            "answers": answers,
            "date": datetime.now().isoformat()
        }).execute()
        return True
    except Exception:
        return False
        
def save_ece_log_db(user_id, hr: float, stress: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("ece_logs").insert({
            "user_id": user_id, 
            "filtered_hr": hr, 
            "gsr_stress": stress,
            "created_at": datetime.now().isoformat()
        }).execute()
        return True
    except Exception:
        return False


# CRITICAL FIX: Removed 'supabase_client' argument to avoid UnhashableParamError
@st.cache_data(ttl=60)
def load_all_user_data(user_id: str): 
    """Loads all user data from Supabase for caching purposes."""
    
    # Access the client directly from session state inside the function
    supabase_client = st.session_state.get("_supabase_client_obj") 
    
    if not supabase_client:
        return {}

    user_data = {}
    
    # 1. Load User Profile
    try:
        res = supabase_client.table("profiles").select("*").eq("id", user_id).single().execute()
        user_data["profile"] = res.data
    except Exception:
        user_data["profile"] = None

    # 2. Load Mood History
    try:
        res = supabase_client.table("mood_logs").select("*").eq("user_id", user_id).order("date", desc=True).limit(50).execute()
        user_data["mood_history"] = res.data
    except Exception:
        user_data["mood_history"] = []

    # 3. Load Journal Entries
    try:
        res = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("date", desc=True).execute()
        user_data["journal_entries"] = res.data
    except Exception:
        user_data["journal_entries"] = []

    # 4. Load ECE Logs (Latest 100 entries)
    try:
        res = supabase_client.table("ece_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(100).execute()
        user_data["ece_logs"] = res.data
    except Exception:
        user_data["ece_logs"] = []
        
    # 5. Load Wellness Check-ins (Latest 5)
    try:
        res = supabase_client.table("wellness_checkins").select("*").eq("user_id", user_id).order("date", desc=True).limit(5).execute()
        user_data["wellness_checkins"] = res.data
    except Exception:
        user_data["wellness_checkins"] = []

    return user_data


# ---------- BUSINESS LOGIC & CORE APP FUNCTIONS ----------

def calculate_phq9_score(answers: dict) -> int:
    """Calculates the total PHQ-9 score from the dictionary of answers."""
    return sum(answers.values())

def calculate_plant_health(mood_history: list, ece_logs: list) -> int:
    """Calculates plant health based on recent mood and stress."""
    
    # 1. Mood Health (Weight: 60%)
    if mood_history:
        # Use only the last 7 unique days of mood data
        df_mood = pd.DataFrame(mood_history)
        df_mood['date'] = pd.to_datetime(df_mood['date']).dt.date
        df_mood = df_mood.drop_duplicates(subset=['date'], keep='first').head(7)
        
        if not df_mood.empty:
            # Normalize mood score (1-10) to health factor (0-1)
            mood_factor = (df_mood['mood'].mean() - 1) / 9.0
            mood_health = 60 * mood_factor
        else:
            mood_health = 40 # Neutral if no data
    else:
        mood_health = 40

    # 2. Stress Health (Weight: 40%)
    if ece_logs:
        df_ece = pd.DataFrame(ece_logs)
        df_ece['stress'] = df_ece['gsr_stress'] # Map gsr_stress to 'stress'
        
        # Calculate daily average stress from ECE logs (last 7 days)
        df_ece['created_at'] = pd.to_datetime(df_ece['created_at'])
        df_ece['date'] = df_ece['created_at'].dt.date
        
        df_daily_stress = df_ece.groupby('date')['stress'].mean().reset_index()
        df_daily_stress = df_daily_stress.sort_values('date', ascending=False).head(7)

        if not df_daily_stress.empty:
            # Stress factor (1.0 - 5.0). Inverse: 5.0 (high stress) -> 0 health
            avg_stress = df_daily_stress['stress'].mean()
            stress_factor = (5.0 - avg_stress) / 4.0 
            stress_health = 40 * stress_factor
        else:
            stress_health = 30 # Neutral if no ECE data
    else:
        stress_health = 30
        
    # Combine health and clamp to 0-100
    total_health = int(mood_health + stress_health)
    return max(0, min(100, total_health))


def logout():
    """Clears all session state variables and logs the user out."""
    st.session_state["logged_in"] = False
    st.session_state["user_id"] = None
    st.session_state["user_email"] = None
    st.session_state["page"] = "Home"
    st.session_state["chat_history"] = [{"role": "system", "content": "You are Harmony, a youth wellness coach focused on CBT and mindfulness. Your tone is supportive, encouraging, and non-judgemental. Keep responses concise and actionable. Do not give medical advice."}]
    # Clear cached data
    load_all_user_data.clear() 
    st.rerun()


# ---------- PAGE FUNCTIONS ----------

def homepage_panel():
    st.title(f"Welcome Back, {st.session_state.get('user_email', 'User').split('@')[0]}! 👋")
    st.markdown("---")
    
    # Fetch Data
    user_data = load_all_user_data(st.session_state["user_id"])
    
    # Update Session State with current data
    st.session_state["mood_history"] = user_data.get("mood_history", [])
    st.session_state["journal_entries"] = user_data.get("journal_entries", [])
    st.session_state["ece_logs"] = user_data.get("ece_logs", [])
    
    # Calculate Metrics
    current_health = calculate_plant_health(st.session_state["mood_history"], st.session_state["ece_logs"])
    st.session_state["plant_health"] = current_health
    
    # Determine Status
    if current_health > 75:
        status_text = "Thriving! Keep up the great work."
        status_color = "#38C982"
    elif current_health > 50:
        status_text = "Healthy. Consistent self-care is key."
        status_color = "#FFC400"
    else:
        status_text = "Needs attention. Try checking in with your Mood or Journal."
        status_color = "#FF5252"

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your HarmonySphere 🌱")
        st.markdown(f"""
            <div style='text-align: center; border: 1px solid {status_color}; padding: 10px; border-radius: 10px;'>
                <h1 style='color: {status_color}; margin-top: 0;'>{current_health}%</h1>
                <p style='margin-bottom: 0;'>{status_text}</p>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state["mood_history"]:
            latest_mood = pd.to_datetime(st.session_state["mood_history"][0]['date']).strftime('%b %d')
            latest_mood_score = st.session_state["mood_history"][0]['mood']
            latest_mood_label = MOOD_EMOJI_MAP.get(latest_mood_score, "N/A")
            st.metric("Latest Mood", f"{latest_mood_label}", delta=f"Logged on {latest_mood}")
        
    with col2:
        st.subheader("Your Daily Focus")
        
        # Display a random quote
        quote = random.choice(QUOTES)
        st.markdown(f"***\"{quote}\"***")

        st.markdown("---")
        st.subheader("Quick Actions")
        
        q_col1, q_col2, q_col3 = st.columns(3)
        if q_col1.button("Mood Check-in", use_container_width=True):
            st.session_state["page"] = "Mood Tracker"
            st.rerun()
            
        if q_col2.button("Start Journal", use_container_width=True):
            st.session_state["page"] = "Mindful Journaling"
            st.rerun()
            
        if q_col3.button("Chat with Harmony", use_container_width=True):
            st.session_state["page"] = "AI Chat"
            st.rerun()
            

def mood_tracker_page():
    # --- Page Title ---
    st.title("Mood Tracker 📈")
    st.subheader("Your Emotional Journey, Visualized")
    
    st.markdown("---")

    # --- Mood Logging Form ---
    with st.form("mood_log_form", clear_on_submit=True):
        st.subheader("Log Your Current Mood")
        
        # Reverse the map for display
        score_options = {v: k for k, v in MOOD_EMOJI_MAP.items()}
        
        mood_selection = st.select_slider(
            'How are you feeling right now?',
            options=list(score_options.keys()),
            value=MOOD_EMOJI_MAP.get(5) # Default to Neutral
        )
        
        note = st.text_area("Optional: Any quick thoughts or context?", max_chars=200)
        
        submitted = st.form_submit_button("Save Mood Log")
        
        if submitted:
            mood_score = score_options[mood_selection]
            user_id = st.session_state["user_id"]
            
            if save_mood_db(user_id, mood_score, note):
                st.success(f"Mood Log Saved! Score: {mood_score} ({mood_selection})")
                
                # Clear cache and reload data after saving
                load_all_user_data.clear()
                st.session_state["page"] = "Home" # Refresh data on Home page
                st.rerun()
            else:
                st.error("Failed to save mood log. Please check your connection.")

    # --- Mood History Chart ---
    if not st.session_state["mood_history"]:
        st.info("No mood logs found yet. Log a mood entry to see your history!")
        return
        
    df_mood = pd.DataFrame(st.session_state["mood_history"])
    
    # Ensure 'date' column is in datetime format for operations
    df_mood['date'] = pd.to_datetime(df_mood['date'])

    st.markdown("---")
    st.subheader("Your Mood History (Last 30 Days)")

    # --- Daily Deduplication (The Fixed Section - previous fix) ---
    
    # 1. Create a temporary column with only the date part (no time)
    df_mood['date_only'] = df_mood['date'].dt.date

    # 2. Sort by date descending and drop duplicates on the 'date_only' column.
    df_mood_daily = (
        df_mood
        .sort_values('date', ascending=False)
        .drop_duplicates(subset=['date_only'], keep='first') 
    )

    # 3. Clean up the DataFrame by removing the temporary column
    df_mood_daily = df_mood_daily.drop(columns=['date_only']) 
    
    # Limit the view to the last 30 unique days
    df_mood_daily = df_mood_daily.head(30)
    
    # Sort again for chart display (oldest to newest)
    df_mood_daily = df_mood_daily.sort_values('date', ascending=True)

    # --- Chart Generation ---
    # Map numerical score to the emoji string for better visualization
    df_mood_daily['mood_label'] = df_mood_daily['mood'].map(MOOD_EMOJI_MAP)
    
    # Create the line chart
    fig = px.line(
        df_mood_daily, 
        x='date', 
        y='mood', 
        text='mood_label',
        title='Daily Emotional Trend',
        labels={'date': 'Date', 'mood': 'Mood Score (1-10)'},
        color_discrete_sequence=["#FF6F91"] 
    )
    
    # Customize layout
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title=None,
        yaxis=dict(
            tickmode='array',
            tickvals=list(MOOD_EMOJI_MAP.keys()),
            ticktext=[f"{v.split(' ')[0]} {k}" for k, v in MOOD_EMOJI_MAP.items()],
            range=[1, 11]
        ),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Latest Mood Entries (Table) ---
    st.markdown("---")
    st.subheader("Detailed Mood Log")
    
    # Prepare data for detailed display (using all logs, not just daily)
    df_mood_detail = pd.DataFrame(st.session_state["mood_history"])
    df_mood_detail['Date & Time'] = pd.to_datetime(df_mood_detail['date']).dt.strftime('%Y-%m-%d %H:%M')
    df_mood_detail['Mood'] = df_mood_detail['mood'].map(MOOD_EMOJI_MAP)
    
    st.dataframe(
        df_mood_detail[['Date & Time', 'Mood', 'note']].rename(columns={'note': 'Note/Context'}),
        hide_index=True,
        use_container_width=True
    )


def mindful_journaling_page():
    st.title("Mindful Journaling ✍️")
    st.subheader("Express, Reflect, and Understand Your Feelings")
    
    st.markdown("---")

    with st.form("journal_entry_form", clear_on_submit=True):
        st.subheader("New Journal Entry")
        entry_text = st.text_area("What's on your mind today? Write for at least a few sentences to get the best analysis.", height=250)
        
        submitted = st.form_submit_button("Analyze & Save Entry")
        
        if submitted:
            if len(entry_text) < 50:
                st.warning("Please write a bit more to allow for meaningful reflection.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    analyzer = SentimentIntensityAnalyzer()
                    sentiment = analyzer.polarity_scores(entry_text)['compound']
                    
                    user_id = st.session_state["user_id"]
                    
                    if save_journal_db(user_id, entry_text, sentiment):
                        st.success("Journal Entry Saved and Analyzed!")
                        
                        # Provide quick feedback
                        st.info(f"Sentiment Score: {sentiment:.2f} (closer to 1.0 is positive, -1.0 is negative)")
                        
                        # Clear cache and reload data
                        load_all_user_data.clear()
                        st.session_state["page"] = "Home" 
                        st.rerun()
                    else:
                        st.error("Failed to save journal entry. Please check your connection.")

    st.markdown("---")
    st.subheader("Recent Entries")
    
    if not st.session_state["journal_entries"]:
        st.info("You haven't made any entries yet. Start writing!")
        return

    df_journal = pd.DataFrame(st.session_state["journal_entries"])
    df_journal['date'] = pd.to_datetime(df_journal['date']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        df_journal[['date', 'entry_text', 'sentiment_score']].rename(columns={
            'date': 'Date', 
            'entry_text': 'Entry Snippet', 
            'sentiment_score': 'Sentiment'
        }).head(5),
        hide_index=True,
        column_config={
            "Entry Snippet": st.column_config.TextColumn(
                "Entry Snippet", 
                width="large",
                help="A snippet of the journal entry."
            )
        },
        use_container_width=True
    )

def wellness_checkin_page():
    st.title("Wellness Check-in (PHQ-9) 📋")
    st.subheader("A brief assessment to track your mental well-being")
    
    st.markdown("---")
    
    phq9_questions = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself—or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead or of hurting yourself in some way"
    ]
    
    with st.form("phq9_form"):
        st.markdown("Over the **last two weeks**, how often have you been bothered by the following problems?")
        st.markdown("*(0 = Not at all; 3 = Nearly every day)*")
        
        answers = {}
        for i, question in enumerate(phq9_questions):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Q{i+1}.** {question}")
            with col2:
                answers[f"q{i+1}"] = st.radio(
                    "Score", 
                    options=list(WELLNESS_OPTIONS.keys()),
                    format_func=lambda x: WELLNESS_OPTIONS[x],
                    index=0,
                    key=f"phq9_q{i+1}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
        
        submitted = st.form_submit_button("Submit Check-in")
        
        if submitted:
            score = calculate_phq9_score(answers)
            st.session_state["phq9_score"] = score # Update session state
            
            user_id = st.session_state["user_id"]
            
            if save_wellness_checkin_db(user_id, score, answers):
                st.success(f"Check-in Complete! Your score is {score}.")
                
                # Interpret the score
                if score <= 4:
                    st.info("Minimal depression: Score suggests you are doing well.")
                elif score <= 9:
                    st.warning("Mild depression: Monitor your symptoms. Consider lifestyle changes.")
                elif score <= 14:
                    st.error("Moderate depression: It may be helpful to talk to a professional.")
                else:
                    st.error("Severe depression: Please seek professional support immediately.")
                
                # Clear cache and reload data
                load_all_user_data.clear()
                st.session_state["page"] = "Home" 
                st.rerun()
            else:
                st.error("Failed to save check-in. Please check your connection.")


def ai_chat_page():
    st.title("Chat with Harmony 🤖")
    st.subheader("Your AI Wellness Coach")
    
    if not st.session_state["_ai_connected"]:
        st.error("Harmony is currently offline. Please ensure your OPENROUTER_API_KEY is configured.")
        return

    ai_client = st.session_state["_ai_client_obj"]
    
    # Display chat history
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask Harmony a question about your well-being..."):
        # Add user message to history
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            full_response = ""
            message_placeholder = st.empty()
            
            try:
                # Use only the last 10 messages for context
                context = st.session_state["chat_history"][-10:]
                
                stream = ai_client.chat.completions.create(
                    model=OPENROUTER_MODEL_NAME,
                    messages=context,
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except APIError as e:
                full_response = f"Sorry, I ran into an API error: {e.status_code}. Please try again later."
                st.error(full_response)
            except Exception as e:
                full_response = f"An unexpected error occurred: {e}"
                st.error(full_response)


        # Add assistant response to history
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})


def wellness_ecosystem_page():
    st.title("Wellness Ecosystem 🌱")
    st.subheader("Watch your digital companion grow based on your self-care.")

    current_health = st.session_state["plant_health"]
    
    # Simple plant visualization based on health score
    if current_health > 90:
        plant_emoji = "🌳"
        status = "Flourishing! A model of harmony."
    elif current_health > 70:
        plant_emoji = "🪴"
        status = "Vibrant and healthy."
    elif current_health > 50:
        plant_emoji = "🌿"
        status = "Steady, but could use more care."
    elif current_health > 25:
        plant_emoji = "🌾"
        status = "Needs immediate attention and self-care."
    else:
        plant_emoji = "🍂"
        status = "Wilting. Please prioritize a check-in."

    st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid #5D54A4; border-radius: 15px;'>
            <h1>{plant_emoji}</h1>
            <h2>Health: {current_health}%</h2>
            <p style='font-size: 1.2em;'>Status: {status}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Health Factors")
    
    st.info("The plant's health is calculated daily based on your mood logs and simulated stress levels.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mood History Score", f"{st.session_state.get('latest_mood_score', 'N/A')}")
        st.progress(st.session_state["plant_health"] / 100.0, text="Overall Health")
        
    with col2:
        st.metric("Latest Stress Level (Sim.)", f"{st.session_state['latest_ece_data']['stress']:.2f} (Lower is better)")
        st.progress(1.0 - (st.session_state['latest_ece_data']['stress'] / 5.0), text="Stress Management")


def mindful_breathing_page():
    st.title("Mindful Breathing 🧘‍♀️")
    st.subheader("Follow the animation for a calming 4-7-8 breath cycle.")
    
    st.markdown("""
        **Technique:**
        1. **Inhale** quietly through your nose for **4** seconds.
        2. **Hold** your breath for **7** seconds.
        3. **Exhale** completely through your mouth, making a whoosh sound, for **8** seconds.
    """)
    st.markdown("---")
    
    st.warning("Imagine a slow, expanding/contracting circle below. Close your eyes and sync your breath to it for 5 cycles.")

    # Simple text-based timer simulation
    if st.button("Start 5-Cycle Timer"):
        timer_placeholder = st.empty()
        for i in range(5):
            timer_placeholder.success(f"Cycle {i+1}/5: 🧘‍♀️ INHALE (4s)...")
            time.sleep(4)
            timer_placeholder.info(f"Cycle {i+1}/5: ✋ HOLD (7s)...")
            time.sleep(7)
            timer_placeholder.error(f"Cycle {i+1}/5: 🌬️ EXHALE (8s)...")
            time.sleep(8)
        timer_placeholder.markdown("### ✨ Session Complete! ✨")
        st.balloons()


def cbt_thought_record_page():
    st.title("CBT Thought Record 📝")
    st.subheader("Challenge automatic negative thoughts (ANTs).")
    
    st.markdown("---")

    with st.form("cbt_form", clear_on_submit=True):
        st.markdown("**Step 1: Situation & Emotion**")
        situation = st.text_area("What was the situation that triggered the feeling?", height=100)
        emotion = st.text_input("What was the main emotion and its intensity (e.g., Anxiety 80%)?")

        st.markdown("**Step 2: Automatic Thought**")
        ant = st.text_area("What was the exact automatic thought that went through your mind?", height=100)
        
        st.markdown("**Step 3: Evidence**")
        evidence_for = st.text_area("Evidence **for** the thought (facts that support it):")
        evidence_against = st.text_area("Evidence **against** the thought (facts that contradict it or offer an alternative view):")
        
        st.markdown("**Step 4: Balanced Thought**")
        balanced = st.text_area("What is a more realistic, balanced, and helpful perspective?")
        
        submitted = st.form_submit_button("Submit Thought Record")
        
        if submitted and balanced:
            st.success("Thought Record Complete! You successfully challenged an Automatic Negative Thought.")
            st.markdown(f"**New Perspective:** *{balanced}*")
            st.markdown("Saving this to your journal history (implicitly).")
        elif submitted:
            st.warning("Please fill out the form completely, especially the Balanced Thought section.")


def journal_analysis_page():
    st.title("Journal Sentiment Analysis 📊")
    st.subheader("Track your emotional trend over time.")

    if not st.session_state["journal_entries"]:
        st.info("No journal entries found. Start a journal entry to view this page.")
        return

    df_journal = pd.DataFrame(st.session_state["journal_entries"])
    df_journal['date'] = pd.to_datetime(df_journal['date'])
    
    st.markdown("---")
    
    # Calculate rolling average sentiment
    df_journal = df_journal.sort_values('date')
    df_journal['rolling_sentiment'] = df_journal['sentiment_score'].rolling(window=5, min_periods=1).mean()
    
    fig = px.line(
        df_journal,
        x='date',
        y='rolling_sentiment',
        title='5-Entry Rolling Average Sentiment',
        labels={'date': 'Date', 'rolling_sentiment': 'Sentiment Score'},
        color_discrete_sequence=["#5D54A4"]
    )
    
    fig.update_layout(
        yaxis_range=[-1.0, 1.0],
        yaxis_title="Sentiment Score (-1.0 to 1.0)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Journal Summary")
    
    avg_sentiment = df_journal['sentiment_score'].mean()
    
    if avg_sentiment > 0.3:
        summary = "Overall positive emotional tone. Keep nurturing this healthy mindset."
    elif avg_sentiment < -0.3:
        summary = "Overall negative emotional tone. It might be time to use the CBT Thought Record or check in with Harmony."
    else:
        summary = "Overall neutral or balanced emotional tone. Pay attention to specific spikes or dips in the chart."
        
    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", delta=summary)


def iot_dashboard_page():
    st.title("IoT Dashboard (Simulated ECE) ⌚")
    st.subheader("Emotional & Cardiovascular Ecosystem Data")
    
    st.markdown("---")
    
    # 1. Generate & Update Data
    if st.button("Generate New ECE Data Point"):
        new_data = generate_simulated_physiological_data()
        st.session_state["latest_ece_data"] = new_data
        
        # Save to DB (optional, but good practice)
        save_ece_log_db(
            st.session_state["user_id"], 
            new_data["hr"], 
            new_data["stress"]
        )
        load_all_user_data.clear() # Clear cache
        st.rerun()

    latest_data = st.session_state["latest_ece_data"]
    
    # 2. Display Latest Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Heart Rate (BPM)", f"{latest_data['hr']}", delta_color="normal")
    col2.metric("Sim. Stress Level", f"{latest_data['stress']:.2f}", delta_color="inverse")
    col3.metric("Last Log Time", datetime.now().strftime("%H:%M:%S"))

    st.markdown("---")

    # 3. Display Raw PPG Waveform
    st.subheader("Raw PPG Signal (Heartbeat Simulation)")
    
    # Create a DataFrame for the raw signal
    df_ppg = pd.DataFrame({
        'time': range(len(latest_data['ppg_raw'])),
        'signal': latest_data['ppg_raw']
    })
    
    fig_ppg = px.line(
        df_ppg, 
        x='time', 
        y='signal', 
        title=f"Simulated PPG Signal at {latest_data['hr']} BPM",
        labels={'signal': 'Voltage/Intensity', 'time': 'Sample Index'},
        color_discrete_sequence=["#FF5252"]
    )
    
    fig_ppg.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_ppg, use_container_width=True)

    # 4. Stress History Chart
    st.markdown("---")
    st.subheader("Stress History (Last 100 Logs)")
    
    if not st.session_state["ece_logs"]:
        st.info("Generate some ECE data points to see a history chart.")
        return
        
    df_ece = pd.DataFrame(st.session_state["ece_logs"])
    df_ece['created_at'] = pd.to_datetime(df_ece['created_at'])
    df_ece = df_ece.sort_values('created_at', ascending=True)
    
    fig_stress = px.line(
        df_ece, 
        x='created_at', 
        y='gsr_stress', 
        title="GSR Stress Trend",
        labels={'created_at': 'Time', 'gsr_stress': 'Stress Level (1.0 to 5.0)'},
        color_discrete_sequence=["#5D54A4"]
    )
    
    fig_stress.update_layout(yaxis_range=[1.0, 5.0], hovermode="x unified")
    st.plotly_chart(fig_stress, use_container_width=True)


def report_summary_page():
    st.title("Report & Summary 📑")
    st.subheader("Your Progress at a Glance")
    
    # Fetch Data
    user_data = load_all_user_data(st.session_state["user_id"])
    mood_history = user_data.get("mood_history", [])
    journal_entries = user_data.get("journal_entries", [])
    wellness_checkins = user_data.get("wellness_checkins", [])
    
    st.markdown("---")

    # --- Section 1: Mood Summary ---
    st.header("1. Emotional Health Summary")
    if mood_history:
        df_mood = pd.DataFrame(mood_history)
        df_mood['date'] = pd.to_datetime(df_mood['date'])
        
        avg_mood = df_mood['mood'].mean()
        
        st.metric("Average Mood Score (Last 50 Logs)", f"{avg_mood:.2f} / 10")
        
        # Chart of Mood Distribution
        mood_counts = df_mood['mood'].value_counts().reset_index()
        mood_counts.columns = ['mood', 'count']
        mood_counts['mood_label'] = mood_counts['mood'].map(MOOD_EMOJI_MAP)

        fig_mood_dist = px.bar(
            mood_counts, 
            x='mood', 
            y='count', 
            text='mood_label', 
            title='Mood Score Distribution',
            color_discrete_sequence=["#FF6F91"]
        )
        fig_mood_dist.update_traces(textposition='outside')
        st.plotly_chart(fig_mood_dist, use_container_width=True)
    else:
        st.info("No mood data to display.")

    st.markdown("---")

    # --- Section 2: Wellness Check-in Summary ---
    st.header("2. PHQ-9 Check-in History")
    if wellness_checkins:
        df_checkins = pd.DataFrame(wellness_checkins)
        df_checkins['date'] = pd.to_datetime(df_checkins['date'])
        df_checkins = df_checkins.sort_values('date', ascending=True)

        fig_phq9 = px.line(
            df_checkins, 
            x='date', 
            y='phq9_score', 
            title='PHQ-9 Score Over Time',
            labels={'phq9_score': 'PHQ-9 Score', 'date': 'Date'},
            color_discrete_sequence=["#FFC400"]
        )
        st.plotly_chart(fig_phq9, use_container_width=True)
        st.metric("Latest PHQ-9 Score", f"{df_checkins.iloc[-1]['phq9_score']}")
    else:
        st.info("No wellness check-in data to display.")


def sidebar_auth():
    """Handles the sidebar login and authentication logic."""
    st.sidebar.title("HarmonySphere 🌿")
    st.sidebar.markdown("---")
    
    if st.session_state.get("logged_in"):
        # Logged In View
        st.sidebar.markdown(f"**Logged In:** {st.session_state['user_email']}")
        
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.subheader("Navigation")
        pages = [
            "Home", 
            "Mood Tracker", 
            "Mindful Journaling", 
            "Wellness Check-in",
            "Wellness Ecosystem", 
            "IoT Dashboard (ECE)",
            "AI Chat",
            "Report & Summary",
            "Mindful Breathing", 
            "CBT Thought Record",
            "Journal Analysis"
        ]
        
        current_selection = st.sidebar.radio(
            "Go to Page:",
            pages,
            index=pages.index(st.session_state["page"]),
            key="page_selector"
        )
        
        # Update page state if selection changes
        if current_selection != st.session_state["page"]:
            st.session_state["page"] = current_selection
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.button("Logout", on_click=logout)
        
    else:
        # Not Logged In View (Login/Register Form)
        with st.sidebar.form("login_form"):
            st.subheader("Login or Register")
            
            email = st.text_input("Email Address", key="auth_email_input").lower().strip()
            
            submitted = st.form_submit_button("Access Dashboard")
            
            if submitted:
                # Store email in session state temporarily for login flow
                st.session_state["temp_email_attempt"] = email
                st.session_state["page"] = "Auth" # Use a temp state to handle redirect
                st.session_state["show_splash"] = False # Hide splash screen
                time.sleep(0.1)
                st.rerun()
                
        st.sidebar.markdown("---")
        if not st.session_state.get("_db_connected"):
            st.sidebar.error("Database connection failed. Running in LOCAL mode.")
        if not st.session_state.get("_ai_connected"):
            st.sidebar.warning("AI Chat is offline. Check OpenRouter key.")


def app_splash_screen():
    """Initial loading screen."""
    st.title("HarmonySphere 🌿")
    st.subheader("Your Digital Youth Wellness Companion")
    st.markdown("---")
    
    st.info("Loading necessary models and data connections. Please wait.")
    
    # Simulate loading time for initial user experience
    time.sleep(2) 
    
    # After loading, transition to the login page
    st.session_state["show_splash"] = False
    st.rerun()


def unauthenticated_home():
    """The central login/landing page."""
    
    # Use the email from the sidebar submission attempt
    email = st.session_state.get("temp_email_attempt")
    submitted = (email is not None)
    
    if submitted:
        st.session_state["temp_email_attempt"] = None # Clear the temp state

        with st.spinner("Checking credentials..."):
            
            if email and "@" in email:
                
                user = None
                db_connected = st.session_state.get("_db_connected")

                # --- 1. Login/Lookup Attempt (Using Admin Client) ---
                if db_connected:
                    user_list = get_user_by_email_db(email) 
                    if user_list:
                        user = user_list[0] # If user exists, 'user' is now set

                # Check 1: If user exists OR we are in LOCAL mode
                if user or db_connected is False: 
                    
                    # --- AUTHENTICATION SUCCESS (Existing User or Local Mode) ---
                    st.session_state["user_id"] = user.get("id") if user else f"local_user_{email.split('@')[0]}"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True

                    # CRITICAL FIX: The function call is fixed here to remove the unhashable argument
                    user_data = load_all_user_data(st.session_state["user_id"]) 
                    
                    # Update metrics and data lists
                    st.session_state["phq9_score"] = user_data.get("phq9_score", 0)
                    st.session_state["mood_history"] = user_data.get("mood_history", [])
                    st.session_state["journal_entries"] = user_data.get("journal_entries", [])
                    st.session_state["ece_logs"] = user_data.get("ece_logs", [])
                    st.session_state["plant_health"] = calculate_plant_health(st.session_state["mood_history"], st.session_state["ece_logs"])
                    
                    st.success("Login successful! Redirecting to dashboard...")
                    time.sleep(1.0) 
                    st.session_state["page"] = "Home"
                    st.rerun()

                else:
                    # --- 2. Registration Attempt (New User) ---
                    if db_connected:
                        st.info("User not found. Attempting to register new user...") 
                        uid = register_user_db(email) 
                        
                        if uid:
                            # Registration Success
                            st.session_state["user_id"] = uid
                            st.session_state["user_email"] = email
                            st.session_state["logged_in"] = True
                            
                            st.success("Registration successful! Welcome to HarmonySphere.")
                            time.sleep(1.0)
                            st.session_state["page"] = "Home"
                            st.rerun()
                        else:
                            # If uid is None, the DB Insert Error should already be visible
                            st.error("Failed to register user in DB. Check secrets or Service Key permissions. See logs for details.")
                    else:
                        st.error("Authentication Failed: User not found and DB is disconnected. Cannot proceed.")
            else:
                st.error("Please enter a valid email address.")
                
    # Display the main marketing content when not submitted
    if not st.session_state.get("logged_in"):
        
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; border: 2px solid #5D54A4; border-radius: 15px; background-color: #F7F7FF;">
            <h1 style="color: #5D54A4; margin-top: 0;">HarmonySphere 🌱</h1>
            <p style="font-size: 1.2em; color: #333;">Integrate Mindful Self-Care with Digital Wellness.</p>
            <h3 style="color: #5D54A4; margin-top: 20px;">Access Your Dashboard</h3>
            <p>Please use the login form on the left sidebar to access the app's features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Remember: HarmonySphere is a support tool, not a substitute for medical advice.")


# ---------- MAIN APPLICATION SETUP ----------

# Set page config and initialize state
st.set_page_config(
    page_title="HarmonySphere Wellness",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .stApp {background-color: #f0f2f6;}
    .st-emotion-cache-1pxx9r4 {padding-top: 30px;} 
    h1, h2, h3 {color: #5D54A4;}
    .stButton>button {background-color: #5D54A4; color: white;}
    /* Fix for sidebar content */
    .st-emotion-cache-6q9sum {padding-top: 1rem;} 
</style>
""", unsafe_allow_html=True)

# 1. Initialize session state, DB, and AI clients
# NOTE: The cache clear is inside this function now.
initialize_session_state()

# 2. Render Sidebar
sidebar_auth()

# 3. Use an empty container to control the main content
app_placeholder = st.empty()

# ---------- MAIN APPLICATION LOGIC (Triple Flow) ----------
with app_placeholder.container():
    
    if st.session_state.get("show_splash"):
        # 1. Show Splash Screen first (blocks other content)
        app_splash_screen()
        
    elif not st.session_state.get("logged_in"):
        # 2. Transition to Centered Login
        unauthenticated_home()

    else:
        # 3. Transition to Authenticated Dashboard
        current_page = st.session_state["page"]
        
        # --- AUTHENTICATED PAGES ---
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