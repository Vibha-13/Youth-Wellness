import streamlit as st
import os
import time
import random
import io
import re
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Lightweight sentiment analyzer cached
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- CONSTANTS ----------
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
    ("Self-Aware", lambda s: len(s["mood_history"]) >= 5 and s["streaks"].get("mood_log", 0) >= 5)
]

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# ---------- CACHING & LAZY SETUP ----------
@st.cache_resource
def setup_analyzer():
    return SentimentIntensityAnalyzer()

# Lazy AI setup — defer heavy import until called
@st.cache_resource(show_spinner=False)
def setup_ai_model(api_key: str):
    """Lazy configure google.generativeai if available and key provided.
       Returns (model_obj or None, boolean ai_available, chat_session or None)
    """
    if not api_key:
        return None, False, None
    try:
        # local import to avoid slowing module load
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # **UPDATED SYSTEM INSTRUCTION FOR EMPATHETIC TONE**
        system_instruction = """
You are 'The Youth Wellness Buddy,' an AI designed for teens. 
Your primary goal is to provide non-judgmental, empathetic, and encouraging support. 
Your personality is warm, slightly informal, and very supportive.

Rules for response style:
1. Always validate the user's feelings first ("That sounds really tough," or "Thanks for sharing that.").
2. Give conversational, longer, and connected responses (2-4 sentences minimum).
3. Encourage the user to share more with open-ended questions (e.g., "What does that feeling feel like in your body?").
4. If they change the subject, address the new topic, but gently check if they want to return to the previous one.
"""
        
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=system_instruction
        )
        
        # Prepare history for chat initialization
        history_for_init = []
        if st.session_state["chat_messages"]:
            import google.generativeai.types as genai_types
            for msg in st.session_state["chat_messages"]:
                role_map = msg.get("role")
                if role_map == "assistant":
                    role_map = "model"
                
                content_text = msg.get("content")
                if content_text:
                    history_for_init.append(
                        genai_types.Content(
                            parts=[genai_types.Part.from_text(content_text)], 
                            role=role_map
                        )
                    )

        chat_session = model.start_chat(history=history_for_init)
        
        # Sync initial welcome message if history is empty
        if not st.session_state["chat_messages"] or st.session_state["chat_messages"][0]["role"] != "assistant":
             st.session_state["chat_messages"] = [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]

        return model, True, chat_session
    except Exception as e:
        # print(f"AI Setup failed: {e}") # for debugging
        return None, False, None

# Lazy supabase client setup (defer import)
@st.cache_resource(show_spinner=False)
def setup_supabase_client(url: str, key: str):
    if not url or not key:
        return None, False
    try:
        from supabase import create_client
        client = create_client(url, key)
        return client, True
    except Exception:
        return None, False

# ---------- Session state defaults ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Initialize AI/DB models in state for the first time
if "_ai_model" not in st.session_state:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    _ai_model_tuple, _ai_available, _chat_session_obj = setup_ai_model(GEMINI_API_KEY)
    st.session_state["_ai_model"] = _ai_model_tuple
    st.session_state["_ai_available"] = _ai_available
    st.session_state["chat_session"] = _chat_session_obj
    
if "_supabase_client_obj" not in st.session_state:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
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

if "phq9_score" not in st.session_state:
    st.session_state["phq9_score"] = None

if "phq9_interpretation" not in st.session_state:
    st.session_state["phq9_interpretation"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]


analyzer = setup_analyzer()

# ---------- Helper functions ----------
def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    # Strip non-ASCII characters and clean whitespace
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def safe_generate(prompt: str, max_tokens: int = 300):
    """
    Generate text via Gemini, using a pre-configured chat session and 
    incorporating the custom, empathetic response logic for key phrases.
    
    NOTE: The chat session must be initialized/updated externally (in the panel)
          to ensure the history is correct.
    """
    
    # **CUSTOM, EMPATHETIC RESPONSE LOGIC**
    prompt_lower = prompt.lower()
    
    # Case 1: User expresses demotivation or sadness
    if any(phrase in prompt_lower for phrase in ["demotivated", "heavy", "don't want to do anything", "feeling down"]):
        return (
            "Thanks for reaching out and sharing that with me. Honestly, **that feeling of demotivation can be really heavy, and it takes a lot of courage just to name it.** I want you to know you're definitely not alone in feeling this way. Before we try to tackle the whole mountain, let's just look at one rock. **Is there one tiny task or thought that feels the heaviest right now?** Sometimes just describing it makes it a little lighter. 🌱"
        )

    # Case 2: User explicitly asks for a break or a joke
    elif "funny" in prompt_lower or "joke" in prompt_lower or "break" in prompt_lower:
        # Check for previous message to maintain context
        previous_topic = "our chat"
        # Since the current prompt is the last message in st.session_state.chat_messages, 
        # we look at the message before it for context.
        if len(st.session_state.chat_messages) > 1 and st.session_state.chat_messages[-2]["role"] == "user":
            previous_prompt = st.session_state.chat_messages[-2]["content"]
            previous_topic = f"what you were sharing about '{previous_prompt[:25]}...'"

        return (
            "I hear you! It sounds like you need a quick reset, and a little humor is a great way to do that. **Okay, here's a silly one that always makes me smile:** Why don't scientists trust atoms? **Because they make up everything!** 😂 I hope that got a small chuckle! **Ready to dive back into** " + previous_topic + ", **or should I keep the jokes coming for a few more minutes?**"
        )
    
    # --- For all other inputs, rely on the detailed AI System Instruction ---
    
    if st.session_state.get("_ai_available") and st.session_state.get("chat_session"):
        chat_session = st.session_state["chat_session"]
        prompt_clean = clean_text_for_ai(prompt)
        try:
            # Note: We use the existing chat session here
            resp = chat_session.send_message(prompt_clean, max_output_tokens=max_tokens)
            return getattr(resp, "text", None) or str(resp)
        except Exception:
            # Fallback to canned replies on API/model failure
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
    # Journal entries
    parts += [e.get("text","") for e in st.session_state["daily_journal"] if e.get("text")]
    # User chat messages
    parts += [m.get("content","") for m in st.session_state["chat_messages"] if m.get("role") == "user" and m.get("content")]
    return " ".join(parts).strip()

def generate_wordcloud_figure_if_possible(text: str):
    if not text or not text.strip():
        return None
    try:
        # lazy import
        from wordcloud import WordCloud
        # Filter out common, less meaningful words
        stopwords = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'it', 'I', 'my', 'me', 'that', 'this', 'for', 'was', 'with'])
        wc = WordCloud(
            width=800, height=400, background_color="white", stopwords=stopwords, 
            max_words=100, contour_width=3, contour_color='steelblue'
        ).generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception:
        return None

# ---------- Supabase helpers (lazy & guarded) ----------
def register_user_db(email: str):
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return None
    try:
        # In a real app, handle password/OAuth securely. Here, we simulate simple sign-up/login.
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
        res = supabase_client.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    supabase_client = st.session_state.get("_supabase_client_obj")
    if not supabase_client:
        return False
    try:
        supabase_client.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def load_journal_db(user_id, supabase_client):
    if not supabase_client:
        return []
    try:
        # Using Supabase's built-in order by created_at
        res = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

# ---------- UI style ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; font-family: 'Poppins', sans-serif; }
    .main .block-container { padding: 2rem 3rem; }
    .card { background-color: #eaf4ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 18px; margin-bottom: 18px; border-left: 5px solid #4a90e2; transition: transform .12s; }
    .card:hover { transform: translateY(-4px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
    .stButton>button { color: #fff; background-color: #4a90e2; border-radius: 8px; padding: 8px 18px; font-weight:600; border: none; }
    .stButton>button:hover { background-color: #357bd9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation (placed here to allow functions to be defined below)
st.sidebar.markdown(f"- AI Status: **{'Connected' if st.session_state.get('_ai_available') else 'Local (fallback)'}**")
st.sidebar.markdown(f"- DB Status: **{'Connected' if st.session_state.get('_db_connected') else 'Not connected'}**")

st.sidebar.header("Navigation")
page_options = {
    "Home": "🏠", 
    "AI Chat": "💬", 
    "Mood Tracker": "📈", 
    "Mindful Journaling": "📝", 
    "Journal Analysis": "📊",
    "Mindful Breathing": "🧘‍♀️", 
    "Wellness Check-in": "🩺",
    "Report & Summary": "📄"
}
st.session_state["page"] = st.sidebar.radio("Go to:", list(page_options.keys()), format_func=lambda x: f"{page_options[x]} {x}")


# ---------- Sidebar: Auth (FIXED: st.experimental_rerun -> st.rerun) ----------
def sidebar_auth():
    st.sidebar.header("Account")
    if not st.session_state.get("logged_in"):
        email = st.sidebar.text_input("Your email", key="login_email")
        if st.sidebar.button("Login / Register"):
            if email:
                user = None
                if st.session_state.get("_db_connected"):
                    user_list = get_user_by_email_db(email)
                    if user_list:
                        user = user_list[0]
                
                if user or st.session_state.get("_db_connected") is False:
                    # Login or Local Login Success
                    st.session_state["user_id"] = user.get("id") if user else "local_user"
                    st.session_state["user_email"] = email
                    st.session_state["logged_in"] = True
                    st.session_state["daily_journal"] = [] # Clear local if successful DB login
                    
                    if user and st.session_state.get("_db_connected"):
                        entries = load_journal_db(st.session_state["user_id"], st.session_state.get("_supabase_client_obj")) or []
                        st.session_state["daily_journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in entries]
                        st.sidebar.success("Logged in and data loaded.")
                    elif st.session_state.get("_db_connected") is False:
                         st.sidebar.info("Logged in locally (no DB).")
                         
                    st.rerun() # FIXED

                else:
                    # Attempt Register if DB connected and user not found
                    uid = register_user_db(email)
                    if uid:
                        st.session_state["user_id"] = uid
                        st.session_state["user_email"] = email
                        st.session_state["logged_in"] = True
                        st.sidebar.success("Registered & logged in.")
                        st.rerun() # FIXED
                    else:
                        st.sidebar.error("Registration failed. Try again or check DB connection.")
            else:
                st.sidebar.warning("Enter an email")
    else:
        st.sidebar.write("Logged in as:")
        st.sidebar.markdown(f"**{st.session_state.get('user_email')}**")
        if st.sidebar.button("Logout"):
            # Clear user-specific entries and reset state
            for key in ["logged_in", "user_id", "user_email", "phq9_score", "phq9_interpretation"]:
                st.session_state[key] = None
            st.session_state["daily_journal"] = [] # Clear local journal
            st.session_state.chat_messages = [{"role": "assistant", "content": "Hello 👋 I’m here to listen. What’s on your mind today?"}]
            # Clear chat session instance to force re-init
            st.session_state["chat_session"] = None
            st.sidebar.info("Logged out.")
            st.rerun() # FIXED

sidebar_auth()


# ---------- Panels ----------
def homepage_panel():
    st.title("Your Wellness Sanctuary 🧠")
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("Daily Inspiration ✨")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{random.choice(QUOTES)}**")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("### Quick actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Start Breathing 🧘‍♀️"):
                st.session_state["page"] = "Mindful Breathing"
                st.rerun() # FIXED
        with c2:
            if st.button("Talk to AI 💬"):
                st.session_state["page"] = "AI Chat"
                st.rerun() # FIXED
        with c3:
            if st.button("Journal 📝"):
                st.session_state["page"] = "Mindful Journaling"
                st.rerun() # FIXED
    with col2:
        st.image("https://images.unsplash.com/photo-1549490349-f06b3e942007?q=80&w=2070&auto=format&fit=crop", caption="Take a moment for yourself")
    st.markdown("---")
    st.header("App Features")
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown("#### Mood Tracker 📈")
        st.markdown("Log quick mood ratings and unlock badges.")
    with f2:
        st.markdown("#### AI Chat 💬")
        st.markdown("A compassionate AI to listen and suggest small exercises.")
    with f3:
        st.markdown("#### Journal & Insights 📊")
        st.markdown("Track progress over time with charts and word clouds.")
    
def mood_tracker_panel():
    st.header("Daily Mood Tracker 📈")
    col1, col2 = st.columns([3,1])
    with col1:
        mood = st.slider("How do you feel right now? (1-11)", 1, 11, 6)
        st.markdown(f"**You chose:** {MOOD_EMOJI_MAP.get(mood, 'N/A')} · **{mood}/11**")
        note = st.text_input("Optional: Add a short note about why you feel this way", key="mood_note_input")
        if st.button("Log Mood", key="log_mood_btn"):
            entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
            st.session_state["mood_history"].append(entry)

            # Streak Logic
            last_date = st.session_state["streaks"].get("last_mood_date")
            today = datetime.now().date()
            last_dt = None
            if last_date:
                try:
                    last_dt = datetime.strptime(last_date, "%Y-%m-%d").date()
                except Exception:
                    last_dt = None

            if last_dt != today:
                yesterday = today - timedelta(days=1)
                if last_dt == yesterday:
                    st.session_state["streaks"]["mood_log"] = st.session_state["streaks"].get("mood_log", 0) + 1
                else:
                    st.session_state["streaks"]["mood_log"] = 1
                st.session_state["streaks"]["last_mood_date"] = today.strftime("%Y-%m-%d")

            st.success("Mood logged. Tiny step, big impact. ✨")

            # Badge check
            for name, rule in BADGE_RULES:
                try:
                    state_subset = {"mood_history": st.session_state["mood_history"], "streaks": st.session_state["streaks"]}
                    if rule(state_subset):
                        if name not in st.session_state["streaks"]["badges"]:
                            st.session_state["streaks"]["badges"].append(name)
                except Exception:
                    continue
            st.rerun() # FIXED

    with col2:
        st.subheader("Badges 🎖️")
        if st.session_state["streaks"]["badges"]:
            for b in st.session_state["streaks"]["badges"]:
                st.markdown(f"**{b}** 🌟")
        else:
            st.markdown("_No badges yet — log a mood to get started!_")

        st.subheader("Streak 🔥")
        st.markdown(f"Consecutive days logging mood: **{st.session_state['streaks'].get('mood_log',0)}**")

    # Plot mood history if exists
    if st.session_state["mood_history"]:
        df = pd.DataFrame(st.session_state["mood_history"]).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date') # Ensure correct plotting order
        fig = px.line(df, x='date', y='mood', title="Mood Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

def ai_chat_panel():
    st.header("AI Chat 💬")
    st.markdown("A compassionate AI buddy to listen. All your messages help the AI understand you better.")

    model = st.session_state.get("_ai_model")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What's on your mind?")
    if prompt:
        # 1. Add user message to display (BEFORE calling AI)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Listening closely..."):
                # 2. Call safe_generate
                ai_response = safe_generate(prompt)
                
                st.markdown(ai_response)
                
                # 3. Add AI response to display history
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                
                # 4. Update the actual chat session instance with the new response
                # This ensures the model's internal history is accurate for the next turn.
                if st.session_state.get("chat_session"):
                    try:
                        import google.generativeai.types as genai_types
                        st.session_state["chat_session"].history.append(
                            genai_types.Content(
                                parts=[genai_types.Part.from_text(ai_response)], 
                                role="model"
                            )
                        )
                    except Exception:
                        pass # Ignore if history update fails, fallback is safe_generate logic
                        
        # Use st.rerun() to force Streamlit to clear the input and display new messages
        st.rerun() # FIXED

def mindful_breathing_panel():
    st.header("Mindful Breathing 🧘‍♀️")
    st.markdown("Follow the prompts: **Inhale (4s) — Hold (4s) — Exhale (6s)**. Try **3 cycles** for a quick reset.")
    
    if "breathing_state" not in st.session_state:
        st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}

    bs = st.session_state["breathing_state"]
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("Start Exercise", key="start_breathing_btn")
    with col_btn2:
        if st.button("Reset", key="reset_breathing_btn") and not bs["running"]:
            st.session_state["breathing_state"] = {"running": False, "start_time": None, "cycles_done": 0}
            st.rerun() # FIXED
            return

    if start_btn and not bs["running"]:
        bs["running"] = True
        bs["start_time"] = time.time()
        bs["cycles_done"] = 0
        st.session_state["breathing_state"] = bs
        st.rerun() # FIXED

    if bs["running"]:
        PHASES = [("Inhale 🌬️", 4.0, "#4a90e2"), ("Hold ⏸️", 4.0, "#357bd9"), ("Exhale 💨", 6.0, "#f39c12")]
        total_cycle_time = sum(p[1] for p in PHASES)
        elapsed = time.time() - (bs["start_time"] or time.time())
        
        cycle_number = int(elapsed // total_cycle_time) + 1
        time_in_cycle = elapsed % total_cycle_time

        if cycle_number > 3:
            bs["running"] = False
            bs["cycles_done"] = 3
            st.session_state["breathing_state"] = bs
            st.success("Exercise complete! You did a great job resetting your mind. Keep an eye out for a new badge! 🌟")
            
            # Badge Awarding
            if "Breathing Master" not in st.session_state["streaks"]["badges"]:
                st.session_state["streaks"]["badges"].append("Breathing Master")
                
            st.rerun() # FIXED
            return

        st.info(f"Cycle {cycle_number} of 3")
        
        phase_start = 0.0
        current_phase_name = ""
        current_phase_color = ""
        
        # Determine current phase and time remaining
        for phase, duration, color in PHASES:
            if time_in_cycle < phase_start + duration:
                time_in_phase = time_in_cycle - phase_start
                progress = min(max(time_in_phase / duration, 0.0), 1.0)
                time_remaining = duration - time_in_phase
                
                current_phase_name = phase
                current_phase_color = color
                
                st.markdown(f"<h2 style='text-align:center;color:{current_phase_color};'>{current_phase_name} ({time_remaining:.1f}s remaining)</h2>", unsafe_allow_html=True)
                st.progress(progress)
                break
            phase_start += duration
        
        # refresh frequently but not blocking
        time.sleep(0.1)
        st.rerun() # FIXED

def mindful_journaling_panel():
    st.header("Mindful Journaling 📝")
    st.markdown("Write freely about your day, your feelings, or anything on your mind. Your words are private.")
    
    journal_text = st.text_area("Today's reflection", height=220, key="journal_text")
    
    col_save, col_info = st.columns([1,2])
    with col_save:
        if st.button("Save Entry", key="save_entry_btn"):
            if journal_text.strip():
                sent = sentiment_compound(journal_text)
                
                # Persist to DB if available, else save local
                date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry_data = {"date": date_str, "text": journal_text, "sentiment": sent}
                
                if st.session_state.get("logged_in") and st.session_state.get("_db_connected") and st.session_state.get("user_id"):
                    ok = save_journal_db(st.session_state.get("user_id"), journal_text, sent)
                    if ok:
                        st.success("Saved to your account on Supabase. Your data is secure! 🔒")
                    else:
                        st.warning("Could not save to DB. Saved locally for now.")
                        st.session_state["daily_journal"].append(entry_data)
                else:
                    st.session_state["daily_journal"].append(entry_data)
                    st.success("Saved locally to this browser session. Log in to save permanently. 💾")
                    
                # Clear text area for new entry
                st.session_state["journal_text"] = "" 
                st.rerun() # FIXED
            else:
                st.warning("Write something you want to save.")
    
    with col_info:
        st.info("Saving locally means the entry will be lost if you clear your browser cache.")
        
    st.markdown("---")
    st.subheader("Recent Entries")
    if st.session_state["daily_journal"]:
        for entry in reversed(st.session_state["daily_journal"][-5:]): # Show last 5
            date = pd.to_datetime(entry['date']).strftime('%Y-%m-%d @ %H:%M')
            sentiment = entry.get('sentiment', 0)
            if sentiment >= 0.05:
                label = "🟢 Positive"
            elif sentiment <= -0.05:
                label = "🔴 Negative"
            else:
                label = "⚫ Neutral"
            st.markdown(f"**{date}** ({label})", help=entry.get('text'))
    else:
        st.markdown("_No entries saved yet._")

def journal_analysis_panel():
    st.header("Journal & Analysis 📊")
    
    # --- Data Collection ---
    all_text = get_all_user_text()
    if not all_text:
        st.info("No journal or chat text yet — start journaling or talking to get insights.")
        return

    entries = []
    # Journal entries
    for e in st.session_state["daily_journal"]:
        entries.append({"date": pd.to_datetime(e["date"]), "compound": e.get("sentiment", 0), "source": "Journal"})
    # User chat entries (approximate time as 'now' for simplicity)
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
             entries.append({"date": datetime.now(), "compound": sentiment_compound(msg["content"]), "source": "Chat"})

    if entries:
        df = pd.DataFrame(entries).sort_values("date")
        df["sentiment_label"] = df["compound"].apply(lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral"))
        
        st.subheader("Sentiment Over Time")
        fig = px.line(df, x="date", y="compound", color="sentiment_label", markers=True,
                      title="Emotional Trend based on Entries (VADER)",
                      color_discrete_map={"Positive":"#2ecc71","Neutral":"#95a5a6","Negative":"#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

    # --- Word Cloud ---
    wc_fig = generate_wordcloud_figure_if_possible(all_text)
    if wc_fig:
        st.subheader("Word Cloud of Thoughts 💭")
        st.pyplot(wc_fig, clear_figure=True)
        st.info("The words that appear largest are those you've used most frequently in your entries and chats.")
    
def wellness_check_in_panel():
    st.header("Wellness Check-in (PHQ-9) 🩺")
    st.markdown("This check-in is a **screening tool and not a diagnosis**. If you're worried about your mental health, consider speaking to a professional.")

    phq_questions = [
        "1. Little interest or pleasure in doing things?",
        "2. Feeling down, depressed, or hopeless?",
        "3. Trouble falling or staying asleep, or sleeping too much?",
        "4. Feeling tired or having little energy?",
        "5. Poor appetite or overeating?",
        "6. Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
        "7. Trouble concentrating on things, such as reading the newspaper or watching television?",
        "8. Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
        "9. Thoughts that you would be better off dead, or of hurting yourself in some way?"
    ]

    scores = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3
    }
    score_labels = list(scores.keys())

    with st.form("phq9_form"):
        answers = {}
        for i, q in enumerate(phq_questions):
            # Use index for key to avoid issues with question text changes
            response = st.radio(q, score_labels, key=f"phq9_q{i}", index=0) 
            answers[q] = response
        submitted = st.form_submit_button("Get My Score")

    if submitted:
        total_score = sum(scores[answers[q]] for q in phq_questions)
        interpretation = ""
        
        # Interpretation ranges based on PHQ-9 guidelines
        if total_score >= 20:
            interpretation = "Severe: A high score suggests severe symptoms. It is **strongly recommended you seek professional help immediately**."
        elif total_score >= 15:
            interpretation = "Moderately Severe: This score suggests moderately severe symptoms. Talking to a counselor or doctor is advisable."
        elif total_score >= 10:
            interpretation = "Moderate: This score suggests moderate symptoms. Consider reaching out to a support system or professional."
        elif total_score >= 5:
            interpretation = "Mild: This score suggests mild symptoms. Practicing self-care and checking in with friends can be helpful."
        else:
            interpretation = "Minimal: This score suggests minimal symptoms. Keep up your positive habits!"

        st.session_state["phq9_score"] = total_score
        st.session_state["phq9_interpretation"] = interpretation
        
        st.subheader(f"Your PHQ-9 Score: {total_score}/27")
        if total_score >= 10:
            st.warning(interpretation)
        else:
            st.success(interpretation)
        
        # Suggest Crisis Support if score is high
        if total_score >= 20:
            st.markdown("### Crisis Support Needed")
            st.error("If you feel unsafe or are in crisis, please visit the **Crisis Support** panel immediately for resources.")
        
        st.rerun()

def personalized_report_panel():
    st.header("Personalized Wellness Summary 📄")
    
    st.subheader("Your Progress at a Glance")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(label="Total Mood Logs", value=len(st.session_state["mood_history"]))
    with c2:
        st.metric(label="Journal Entries", value=len(st.session_state["daily_journal"]))
    with c3:
        st.metric(label="Current Streak 🔥", value=st.session_state['streaks'].get('mood_log',0))
        
    st.markdown("---")
    
    st.subheader("PHQ-9 Check-in Summary")
    if st.session_state.get("phq9_score") is not None:
        st.markdown(f"**Latest Score:** {st.session_state['phq9_score']}/27")
        st.markdown(f"**Interpretation:** {st.session_state['phq9_interpretation']}")
    else:
        st.info("No wellness check-in completed yet. Try the **Wellness Check-in** panel.")

    st.markdown("---")
    
    st.subheader("Badges Earned 🎖️")
    if st.session_state["streaks"]["badges"]:
        badge_str = ", ".join(f"**{b}**" for b in st.session_state["streaks"]["badges"])
        st.markdown(badge_str)
    else:
        st.info("Keep logging your mood and journaling to earn your first badge!")

def crisis_support_panel():
    st.header("Crisis Support 🆘")
    st.error("If you are in immediate danger, please call your local emergency services (e.g., 911 in the US).")
    st.markdown("---")
    st.subheader("Suicide & Crisis Resources")
    st.markdown("""
    - **In the US/Canada:** Call or Text **988** (Suicide & Crisis Lifeline)
    - **Crisis Text Line:** Text **HOME** to **741741** (US/Canada) or **85258** (UK)
    - **The Trevor Project:** Call **1-866-488-7386** (for LGBTQ youth)
    - **International:** Visit the **International Association for Suicide Prevention** website to find a crisis center in your country.
    
    **Remember: You are not alone. There is always help available.**
    """)


# ---------- Page Router (uses st.rerun()) ----------
if st.session_state["page"] == "Home":
    homepage_panel()
elif st.session_state["page"] == "AI Chat":
    ai_chat_panel()
elif st.session_state["page"] == "Mood Tracker":
    mood_tracker_panel()
elif st.session_state["page"] == "Mindful Journaling":
    mindful_journaling_panel()
elif st.session_state["page"] == "Journal Analysis":
    journal_analysis_panel()
elif st.session_state["page"] == "Mindful Breathing":
    mindful_breathing_panel()
elif st.session_state["page"] == "Wellness Check-in":
    wellness_check_in_panel()
elif st.session_state["page"] == "Report & Summary":
    personalized_report_panel()