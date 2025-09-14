import streamlit as st
import time, random, io, re, json, tempfile, os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Optional AI & TTS
try:
    import google.generativeai as genai
except:
    genai = None
try:
    import sounddevice as sd
    import wavio
except:
    sd = None
    wavio = None
try:
    import pyttsx3
except:
    pyttsx3 = None
try:
    from supabase import create_client
except:
    create_client = None
try:
    from reportlab.pdfgen import canvas as pdf_canvas
except:
    pdf_canvas = None

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Wellness Companion", page_icon="🧠", layout="wide")

# ---------- STATE ----------
for key in ["messages", "call_history", "daily_journal", "mood_history"]:
    if key not in st.session_state: st.session_state[key] = []
if "streaks" not in st.session_state: st.session_state["streaks"] = {"mood_log":0,"last_mood_date":None,"badges":[]}
if "breath_running" not in st.session_state: st.session_state["breath_running"] = False

analyzer = SentimentIntensityAnalyzer()
MOOD_EMOJI_MAP = {1:"😭",2:"😢",3:"😔",4:"😕",5:"😐",6:"🙂",7:"😊",8:"😄",9:"🤩",10:"🥳"}
QUOTES = ["You are stronger than you think. 💪","Even small steps count. 🌱","Breathe. You are doing your best. 🌬️","Progress, not perfection. Tiny steps add up."]
BADGE_RULES = [("Getting Started", lambda s: len(s["mood_history"]) >= 1),("Weekly Streak: 3", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 3),("Consistent 7", lambda s: s.get("streaks", {}).get("mood_log", 0) >= 7)]

# ---------- AI CONFIG ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
ai_available = False
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        ai_available = True
    except:
        ai_available = False

# ---------- HELPERS ----------
def now_ts(): return time.time()
def clean_text(text): return re.sub(r"[^\x00-\x7F]+"," ",text).strip() if text else ""
def safe_generate(prompt,max_tokens=300):
    if ai_available:
        try: return model.generate_content(clean_text(prompt)).text
        except: pass
    return random.choice(["Thanks for sharing. I hear you — tell me more?","I’m here. Describe one small thing that feels heavy.","We can try a 1-minute breathing exercise together."])
def sentiment(text): return analyzer.polarity_scores(text)["compound"]
def get_all_user_text():
    return " ".join([e.get("text","") for e in st.session_state["daily_journal"]] + [m.get("content","") for m in st.session_state["messages"] if m.get("role")=="user"] + [c.get("text","") for c in st.session_state["call_history"] if c.get("speaker")=="User"]).strip()

def generate_wordcloud_figure(text):
    if not text.strip(): return None
    wc = WordCloud(width=800,height=400,background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def speak_text(text):
    if pyttsx3:
        try: engine = pyttsx3.init(); engine.say(text); engine.runAndWait(); return
        except: pass

def browser_tts(text):
    try:
        components.html(f"""
            <script>
            const utter = new SpeechSynthesisUtterance({{'text':'{text}'}}.text);
            utter.rate=1.0; window.speechSynthesis.cancel(); window.speechSynthesis.speak(utter);
            </script>
        """,height=0)
        return True
    except: return False

def speak_any(text):
    if browser_tts(text): return
    speak_text(text)

# ---------- PANELS ----------
def homepage():
    st.title("Your Wellness Sanctuary")
    st.markdown(f"**{random.choice(QUOTES)}**")
    c1,c2,c3 = st.columns(3)
    if c1.button("Start Breathing"): st.session_state['page']='breathing'; st.rerun()
    if c2.button("Talk to AI"): st.session_state['page']='chat'; st.rerun()
    if c3.button("Journal"): st.session_state['page']='journaling'; st.rerun()

def mood_panel():
    mood = st.slider("How do you feel?",1,10,6); note = st.text_input("Add a note")
    if st.button("Log Mood"):
        entry = {"date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"mood":mood,"note":note}
        st.session_state["mood_history"].append(entry)
        last_date = st.session_state["streaks"].get("last_mood_date")
        today = datetime.now().date()
        last_dt = datetime.strptime(last_date, "%Y-%m-%d").date() if last_date else None
        if last_dt == today: pass
        else: st.session_state["streaks"]["mood_log"] = st.session_state["streaks"].get("mood_log",0)+1; st.session_state["streaks"]["last_mood_date"] = today.strftime("%Y-%m-%d")
        st.success("Mood logged.")
        for name,rule in BADGE_RULES:
            if rule({"mood_history":st.session_state["mood_history"],"streaks":st.session_state["streaks"]}):
                if name not in st.session_state["streaks"]["badges"]: st.session_state["streaks"]["badges"].append(name)
        st.rerun()
    st.write([f"{MOOD_EMOJI_MAP[e['mood']]} {e['mood']}/10 {e.get('note','')}" for e in st.session_state["mood_history"]])

def ai_chat_panel():
    for m in st.session_state["messages"]: st.markdown(f"**{m['role']}**: {m['content']}")
    if prompt:=st.chat_input("What's on your mind?"):
        st.session_state["messages"].append({"role":"user","content":prompt,"ts":now_ts()})
        resp = safe_generate(prompt)
        st.session_state["messages"].append({"role":"assistant","content":resp,"ts":now_ts()})
        st.rerun()

def journaling_panel():
    text = st.text_area("Today's reflection",height=200,key='journal_text')
    if st.button("Save Entry") and text.strip():
        sent = sentiment(text)
        st.session_state["daily_journal"].append({"date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"text":text,"sentiment":sent})
        st.success("Saved.")
        st.rerun()

def analytics_panel():
    all_text = get_all_user_text()
    if not all_text: st.info("No data yet."); return
    df = pd.DataFrame(st.session_state["daily_journal"])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x>=0.05 else ('Negative' if x<=-0.05 else 'Neutral'))
        fig = px.line(df,x='date',y='sentiment',color='sentiment_label',markers=True,color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'})
        st.plotly_chart(fig,use_container_width=True)
    wc_fig = generate_wordcloud_figure(all_text)
    if wc_fig: st.pyplot(wc_fig,clear_figure=True)

def main():
    st.sidebar.title("Navigation")
    pages = {"Home":homepage,"Mood Tracker":mood_panel,"AI Chat":ai_chat_panel,"Journaling":journaling_panel,"Analytics":analytics_panel}
    page = st.sidebar.radio("Go to:",list(pages.keys()))
    pages[page]()

if __name__=="__main__": main()
