import streamlit as st
import os
import time
import random
import io
import pyttsx3
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import sounddevice as sd
import wavio
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load API key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-pro')
        st.sidebar.markdown("Status: **AI API Connected** 🟢")
    except Exception as e:
        st.sidebar.markdown("Status: **AI API Failed** 🔴")
        st.error(f"API Configuration Error: {e}")
else:
    st.sidebar.markdown("Status: **Local Demo Mode** 🟠")
    st.warning("No API key found. The app is running in local demo mode. AI responses are simulated.")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Wellness Companion",
    page_icon="🧠",
    layout="wide",
)

# --- Motivational Quotes Carousel ---
QUOTES = [
    "You are stronger than you think. 💪",
    "Even small steps count. 🌱",
    "Your feelings are valid. 💛",
    "Breathe. You are doing your best. 🌬️",
    "Every day is a new beginning. 🌞"
]

def random_quote():
    return random.choice(QUOTES)

# --- Session State Management ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'transcription_text' not in st.session_state:
    st.session_state['transcription_text'] = ""
if 'sentiment_scores' not in st.session_state:
    st.session_state['sentiment_scores'] = []
if 'analysis_text' not in st.session_state:
    st.session_state['analysis_text'] = ""
if 'call_history' not in st.session_state:
    st.session_state['call_history'] = []
if 'mood_history' not in st.session_state:
    st.session_state['mood_history'] = []
if 'daily_journal' not in st.session_state:
    st.session_state['daily_journal'] = []

# --- Core Functions ---
def get_ai_response(prompt_messages):
    analyzer = SentimentIntensityAnalyzer()
    last_user_message = prompt_messages[-1]['content']
    sentiment_score = analyzer.polarity_scores(last_user_message)['compound']

    # --- Use API if available ---
    if GEMINI_API_KEY:
        try:
            response = model.generate_content(last_user_message)
            ai_reply = response.text
        except:
            ai_reply = fallback_ai_response(last_user_message, sentiment_score)
    else:
        ai_reply = fallback_ai_response(last_user_message, sentiment_score)

    return ai_reply

def fallback_ai_response(user_text, sentiment_score):
    # Local fallback responses
    user_text = user_text.lower()
    if "sad" in user_text or "depressed" in user_text:
        reply = "I hear the heaviness in your words. It's okay to feel this way. 🌧️ What’s one small thing that could bring comfort right now?"
    elif "anxious" in user_text or "panic" in user_text:
        reply = "Take a deep breath with me. I'm here. Can you describe what's making you feel this way? 🌬️"
    elif "happy" in user_text or "good" in user_text:
        reply = "That's wonderful! 😄 What's one thing you're grateful for today?"
    elif "alone" in user_text or "lonely" in user_text:
        reply = "You're not alone. I'm here to listen. 💛 Would you like to talk about what's on your mind?"
    else:
        responses = [
            "I hear you. What’s on your mind? 💭",
            "That sounds challenging. Can you tell me more? 🫂",
            "Thank you for sharing. I'm here to listen. 🧠",
            "Your feelings are valid. What happened next? 🌱"
        ]
        reply = random.choice(responses)

    if sentiment_score < -0.5:
        reply += " Remember, reaching out to a professional can be a helpful step. You don’t have to carry this alone. 🌟"

    return reply

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def analyze_all_sentiment(history, journal):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_list = []

    all_entries = []
    for entry in history:
        all_entries.append({'text': entry['text'], 'date': datetime.fromtimestamp(entry['timestamp'])})
    for entry in journal:
        all_entries.append({'text': entry['text'], 'date': datetime.strptime(entry['date'], "%Y-%m-%d %H:%M:%S")})

    all_entries.sort(key=lambda x: x['date'])

    if not all_entries:
        return None, "No data to analyze yet."

    for entry in all_entries:
        sentiment = analyzer.polarity_scores(entry['text'])
        if sentiment['compound'] >= 0.05:
            color = 'Positive'
        elif sentiment['compound'] <= -0.05:
            color = 'Negative'
        else:
            color = 'Neutral'

        sentiment_list.append({'compound': sentiment['compound'], 'sentiment_color': color, 'date': entry['date']})

    df = pd.DataFrame(sentiment_list)
    return df, "Analysis complete."

def generate_wordcloud(text):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        return fig
    return None

def record_audio(duration=5, fs=44100):
    st.info("Recording... speak now!")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True, dtype='int16')
    st.success("Recording complete!")
    audio_file_bytes = io.BytesIO()
    wavio.write(audio_file_bytes, audio_data, fs, sampwidth=2)
    audio_file_bytes.seek(0)
    return audio_file_bytes

def transcribe_audio(audio_bytes):
    # Placeholder for API STT or local transcription
    dummy_responses = [
        "This is a demo transcription. I'm hearing you clearly. 🌟 What's on your mind?",
        "It sounds like you have a lot to say. Please continue. 🫂",
        "Thank you for speaking. I'm here to listen. 💛"
    ]
    return random.choice(dummy_responses)

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        st.write("Voice output is not available.")

# --- Emoji Mood Mapping ---
MOOD_EMOJIS = ["😔","😐","🙂","😊","😄","🤩","🥳","💪","🌞","✨"]

def mood_to_emoji(mood_value):
    return MOOD_EMOJIS[min(mood_value-1, len(MOOD_EMOJIS)-1)]

# --- Main Pages ---
def homepage():
    st.title("Welcome to AI Wellness Companion 🧠")
    st.markdown(random_quote())
    st.image("https://images.unsplash.com/photo-1510525008061-f09c259d6820?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
    
    st.header("Daily Mood Tracker")
    mood_value = st.slider("Rate your mood today:", 1, 10, 5)
    if st.button("Log Mood"):
        st.session_state.mood_history.append({'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'mood': mood_value})
        st.success(f"Mood logged! {mood_to_emoji(mood_value)}")
    
    if st.session_state.mood_history:
        mood_df = pd.DataFrame(st.session_state.mood_history)
        mood_df['emoji'] = mood_df['mood'].apply(mood_to_emoji)
        fig = px.line(mood_df, x='date', y='mood', text='emoji', markers=True, title="Mood Over Time")
        st.plotly_chart(fig)

def ai_doc_chat():
    st.title("AI Doc Chat 💬")
    st.markdown("Start a conversation with your AI companion.")
    
    for msg in st.session_state.messages:
        role = msg["role"]
        st.chat_message(role).markdown(msg["content"])
    
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role":"user", "content":prompt})
        with st.chat_message("assistant"):
            response = get_ai_response(st.session_state.messages)
            st.markdown(response)
            st.session_state.messages.append({"role":"assistant", "content":response})

def call_session():
    st.title("Call Session 🎤")
    st.warning("Record and receive AI reply. Not a live call.")
    
    if st.button("Start Recording"):
        audio_file = record_audio(duration=15)
        st.session_state.transcription_text = transcribe_audio(audio_file)
        st.session_state.call_history.append({"speaker":"User","text":st.session_state.transcription_text,"timestamp":time.time()})
    
    if st.session_state.transcription_text:
        st.write("You said:")
        st.info(st.session_state.transcription_text)
        st.session_state.messages.append({"role":"user","content":st.session_state.transcription_text})
        
        ai_response = get_ai_response(st.session_state.messages)
        st.session_state.call_history.append({"speaker":"AI","text":ai_response,"timestamp":time.time()})
        st.chat_message("assistant").write(ai_response)
        speak_text(ai_response)
        st.session_state.transcription_text = ""

def journal_and_analysis():
    st.title("Journal & Analysis 📓")
    journal_entry = st.text_area("Write your thoughts:", height=200)
    if st.button("Save Entry"):
        if journal_entry:
            score = get_sentiment(journal_entry)['compound']
            st.session_state.daily_journal.append({"date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text":journal_entry, "sentiment":score})
            st.success("Journal saved!")
    
    all_text = " ".join([e['text'] for e in st.session_state.call_history if e['speaker']=="User"]) + " ".join([e['text'] for e in st.session_state.daily_journal])
    df, status = analyze_all_sentiment(st.session_state.call_history, st.session_state.daily_journal)
    if status == "Analysis complete.":
        fig = px.line(df, x='date', y='compound', color='sentiment_color', markers=True, title="Sentiment Over Time", color_discrete_map={'Positive':'green','Neutral':'yellow','Negative':'red'})
        st.plotly_chart(fig)
        wc_fig = generate_wordcloud(all_text)
        st.pyplot(wc_fig)

# --- Sidebar Navigation ---
page = st.sidebar.radio("Go to:", ("Home","AI Doc Chat","Call Session","Journal & Analysis"))
if page=="Home":
    homepage()
elif page=="AI Doc Chat":
    ai_doc_chat()
elif page=="Call Session":
    call_session()
elif page=="Journal & Analysis":
    journal_and_analysis()
