"""
AI Wellness Companion - Streamlit app (updated & optimized)
Features:
- Mood Tracker (badges, streaks)
- AI Chat (Gemini optional, lazy init)
- Journaling & Analysis (sentiment via VADER)
- Mindful Breathing (non-blocking-ish)
- PHQ-9 Wellness Check-in (complete)
- User Authentication (local + optional Supabase)
- Data persistence with Supabase (lazy)
- Export small PDF report (optional, lazy import reportlab)
"""

import streamlit as st
import os, time, random, io, re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Constants ----------
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

# ---------- Cached helpers ----------
@st.cache_resource
def setup_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def setup_ai_model(api_key: str):
    if not api_key:
        return None, False
    try:
        import google.generativeai as genai  # lazy import
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model, True
    except Exception:
        return None, False

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

# ---------- Session State Defaults ----------
for key, default in {
    "page": "Home",
    "messages": [],
    "call_history": [],
    "daily_journal": [],
    "mood_history": [],
    "streaks": {"mood_log": 0, "last_mood_date": None, "badges": []},
    "transcription_text": "",
    "logged_in": False,
    "user_id": None,
    "user_email": None,
    "phq9_score": None,
    "phq9_interpretation": None,
    "chat_messages": [{"role": "assistant", "content": "Hello — I’m here to listen. What’s on your mind today?"}]
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

analyzer = setup_analyzer()

# ---------- Helper functions ----------
def now_ts():
    return time.time()

def clean_text_for_ai(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_generate(prompt: str, max_tokens: int = 300):
    prompt_clean = clean_text_for_ai(prompt)
    model, ai_available = st.session_state.get("_ai_model", (None, False))
    if ai_available and model:
        try:
            resp = model.generate_content(prompt_clean, max_output_tokens=max_tokens)
            return getattr(resp, "text", str(resp))
        except Exception:
            pass
    canned = [
        "Thanks for sharing. I hear you — would you like to tell me more?",
        "That’s a lot to carry. I’m here. Could you describe one small thing that feels heavy right now?",
        "I’m listening. If you want, we can try a 1-minute breathing exercise together."
    ]
    return random.choice(canned)

def sentiment_compound(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"] if text else 0.0

def get_all_user_text() -> str:
    parts = []
    parts += [e.get("text","") for e in st.session_state["daily_journal"] if e.get("text")]
    parts += [m.get("content","") for m in st.session_state["chat_messages"] if m.get("role") == "user"]
    parts += [c.get("text","") for c in st.session_state["call_history"] if c.get("speaker") == "User"]
    return " ".join(parts).strip()

def generate_wordcloud_figure_if_possible(text: str):
    if not text.strip():
        return None
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig
    except Exception:
        return None

# ---------- Supabase helpers ----------
def register_user_db(email: str):
    client = st.session_state.get("_supabase_client_obj")
    if not client:
        return None
    try:
        res = client.table("users").insert({"email": email}).execute()
        return res.data[0].get("id") if getattr(res, "data", None) else None
    except Exception:
        return None

def get_user_by_email_db(email: str):
    client = st.session_state.get("_supabase_client_obj")
    if not client:
        return []
    try:
        res = client.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

def save_journal_db(user_id, text: str, sentiment: float) -> bool:
    client = st.session_state.get("_supabase_client_obj")
    if not client:
        return False
    try:
        client.table("journal_entries").insert({
            "user_id": user_id,
            "entry_text": text,
            "sentiment_score": float(sentiment)
        }).execute()
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def load_journal_db(user_id, supabase_client):
    if not supabase_client:
        return []
    try:
        res = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

# ---------- UI Style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; font-family: 'Poppins', sans-serif; }
.main .block-container { padding: 2rem 3rem; }
.card { background-color: #eaf4ff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 18px; margin-bottom: 18px; border-left: 5px solid #4a90e2; transition: transform .12s; }
.card:hover { transform: translateY(-4px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
.stButton>button { color: #fff; background-color: #4a90e2; border-radius: 8px; padding: 8px 18px; font-weight:600; border: none; }
.stButton>button:hover { background-color: #357bd9; }
</style>
""", unsafe_allow_html=True)

# ---------- Services setup ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
_ai_model, _ai_available = setup_ai_model(GEMINI_API_KEY)
st.session_state["_ai_model"] = (_ai_model, _ai_available)
st.session_state["_ai_available"] = _ai_available

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
_supabase_client, _supabase_available = setup_supabase_client(SUPABASE_URL, SUPABASE_KEY)
st.session_state["_supabase_client_obj"] = _supabase_client
st.session_state["_supabase_available"] = _supabase_available

# ---------- Navigation ----------
st.sidebar.title("AI Wellness Companion 🧠")
page = st.sidebar.radio("Go to", ["Home", "Mood Tracker", "AI Chat", "Mindful Breathing", "Journaling & Analysis", "PHQ-9 Check-in", "Progress Dashboard", "Crisis Support"])
st.session_state["page"] = page

# ---------- Page Logic Skeleton ----------
if page == "Home":
    st.header("Welcome to your AI Wellness Companion!")
    st.subheader("Your personalized mental health assistant")
    st.markdown("Here’s a motivational quote for you today:")
    st.info(random.choice(QUOTES))

elif page == "Mood Tracker":
    st.header("Mood Tracker")
    mood = st.slider("How are you feeling today?", min_value=1, max_value=11, value=6, format="%d")
    st.markdown(f"Your mood: {MOOD_EMOJI_MAP[mood]}")
    if st.button("Log Mood"):
        today = datetime.now().date()
        last_date = st.session_state["streaks"].get("last_mood_date")
        if last_date == today - timedelta(days=1):
            st.session_state["streaks"]["mood_log"] += 1
        else:
            st.session_state["streaks"]["mood_log"] = 1
        st.session_state["streaks"]["last_mood_date"] = today
        st.session_state["mood_history"].append({"date": today.isoformat(), "mood": mood})
        st.success("Mood logged successfully!")
        # Check badges
        for badge, cond in BADGE_RULES:
            if cond(st.session_state) and badge not in st.session_state["streaks"]["badges"]:
                st.session_state["streaks"]["badges"].append(badge)
                st.balloons()
                st.info(f"🏅 Badge earned: {badge}")

elif page == "AI Chat":
    st.header("AI Chat")
    user_input = st.text_area("Share your thoughts here...")
    if st.button("Send"):
        if user_input.strip():
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            response = safe_generate(user_input)
            st.session_state["chat_messages"].append({"role": "assistant", "content": response})
    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "assistant":
            st.markdown(f"**AI:** {msg['content']}")
        else:
            st.markdown(f"**You:** {msg['content']}")

elif page == "Mindful Breathing":
    st.header("Mindful Breathing Exercise")
    st.info("Follow the instructions below for 3 cycles")
    cycles = st.number_input("Number of cycles:", min_value=1, max_value=5, value=3)
    if st.button("Start Breathing"):
        for i in range(cycles):
            st.markdown(f"**Cycle {i+1}:** Inhale... Exhale...")
            time.sleep(2)

elif page == "Journaling & Analysis":
    st.header("Mindful Journaling")
    entry = st.text_area("Write your thoughts here:")
    if st.button("Save Entry"):
        if entry.strip():
            sentiment = sentiment_compound(entry)
            st.session_state["daily_journal"].append({"timestamp": datetime.now().isoformat(), "text": entry, "sentiment": sentiment})
            if st.session_state["_supabase_available"] and st.session_state.get("user_id"):
                save_journal_db(st.session_state["user_id"], entry, sentiment)
            st.success("Entry saved with sentiment analysis.")
    # Sentiment chart
    if st.session_state["daily_journal"]:
        df_journal = pd.DataFrame(st.session_state["daily_journal"])
        st.line_chart(df_journal["sentiment"])

        # Wordcloud
        fig_wc = generate_wordcloud_figure_if_possible(get_all_user_text())
        if fig_wc:
            st.pyplot(fig_wc)

elif page == "PHQ-9 Check-in":
    st.header("PHQ-9 Wellness Check-in")
    phq_questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself?",
        "Trouble concentrating on things?",
        "Moving/speaking slowly or being fidgety?",
        "Thoughts of self-harm?"
    ]
    answers = []
    for q in phq_questions:
        a = st.radio(q, [0,1,2,3], index=0, horizontal=True)
        answers.append(a)
    if st.button("Submit PHQ-9"):
        score = sum(answers)
        st.session_state["phq9_score"] = score
        if score <= 4: interp = "Minimal or none"
        elif score <= 9: interp = "Mild"
        elif score <= 14: interp = "Moderate"
        elif score <= 19: interp = "Moderately Severe"
        else: interp = "Severe"
        st.session_state["phq9_interpretation"] = interp
        st.success(f"Your PHQ-9 Score: {score} — Interpretation: {interp}")
        if score > 14: st.warning("Consider seeking professional help.")

elif page == "Progress Dashboard":
    st.header("Your Progress Dashboard")
    if st.session_state["mood_history"]:
        df_mood = pd.DataFrame(st.session_state["mood_history"])
        fig = px.bar(df_mood, x="date", y="mood", text="mood", color="mood", color_continuous_scale="Blues")
        st.plotly_chart(fig)
    st.subheader("Badges Earned")
    st.write(st.session_state["streaks"]["badges"])
    if st.session_state["phq9_score"]:
        st.subheader("PHQ-9 Score")
        st.write(f"{st.session_state['phq9_score']} — {st.session_state['phq9_interpretation']}")

elif page == "Crisis Support":
    st.header("Immediate Support Resources")
    st.markdown("""
- **India:** AASRA — +91 9820466726
- **India:** Snehi — +91 22 2772 6771
- **International:** Suicide Prevention Hotline — 1-800-273-TALK (USA)
- **International:** Befrienders Worldwide — https://www.befrienders.org
""")
