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

# --- Custom Styling & Theme ---
st.set_page_config(
    page_title="AI Wellness Companion",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: #0d1117;
        color: #c9d1d9;
    }
    .st-emotion-cache-18ni2cp {
        background-color: #161b22;
        border-radius: 10px;
    }
    .st-emotion-cache-16p649c {
        border: 2px solid #30363d;
        border-radius: 10px;
    }
    .st-emotion-cache-h5h9p4 {
        color: #ffffff;
        background-color: #1f6feb;
        border-radius: 5px;
        border: none;
    }
    .st-emotion-cache-1av55r7 {
        border-color: #30363d;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Navigation")

# --- Load API Key and Configure Model ---
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
    st.warning("No API key found. The app is running in local demo mode. To enable the AI API, add your key to a .env file.")


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
    """
    Attempts to get a response from an AI API,
    falling back to a local response if the API call fails.
    """
    analyzer = SentimentIntensityAnalyzer()
    last_user_message = prompt_messages[-1]['content']
    sentiment_score = analyzer.polarity_scores(last_user_message)['compound']

    if GEMINI_API_KEY:
        try:
            response = model.generate_content(last_user_message)
            return response.text
        except Exception as e:
            st.warning("AI API failed. Using local model for now.")

    # --- Local Fallback Logic (if API fails or is not configured) ---
    last_user_message = last_user_message.lower()
    if "sad" in last_user_message or "depressed" in last_user_message:
        ai_reply = "I hear the heaviness in your words. It's okay to feel this way. What is one small thing that could bring you a bit of comfort right now?"
    elif "anxious" in last_user_message or "stressed" in last_user_message or "panic" in last_user_message:
        ai_reply = "Take a deep breath with me. I'm here. Can you describe what is making you feel this way?"
    elif "happy" in last_user_message or "good" in last_user_message or "great" in last_user_message:
        ai_reply = "That's wonderful to hear! I'm happy for you. What's one thing you're most grateful for from today?"
    elif "alone" in last_user_message or "lonely" in last_user_message:
        ai_reply = "You're not alone. I'm here to listen. Would you like to talk about what's been on your mind?"
    else:
        responses = [
            "I hear you. What's on your mind?",
            "That sounds challenging. Can you tell me more?",
            "Thank you for sharing. I'm here to listen.",
            "Your feelings are valid. What happened next?",
        ]
        ai_reply = random.choice(responses)
        
    if sentiment_score < -0.5:
        ai_reply += " Remember, if you're going through a lot, reaching out to a professional is a great step. You don't have to carry this alone."
        
    return ai_reply

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
            sentiment_color = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_color = 'Negative'
        else:
            sentiment_color = 'Neutral'

        sentiment_list.append({
            'compound': sentiment['compound'],
            'sentiment_color': sentiment_color,
            'date': entry['date']
        })

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
    dummy_responses = [
        "This is a demo transcription. I'm hearing you clearly. What's on your mind?",
        "This is a demo transcription. It sounds like you have a lot to say. Please continue.",
        "This is a demo transcription. Thank you for speaking. I'm here to listen.",
    ]
    return random.choice(dummy_responses)

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Local TTS failed. Error: {e}")
        st.write("Voice output is not available.")

# --- Main app pages ---
def homepage():
    st.title("Welcome to your AI Wellness Companion")
    st.markdown("Your journey to a healthier mind starts here.")
    st.image("https://images.unsplash.com/photo-1510525008061-f09c259d6820?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
    st.markdown("This companion is designed to provide a safe, private space to talk about your thoughts and feelings. Use the navigation on the sidebar to explore different features.")

    st.header("How It Works")
    st.markdown("""
        1. **AI Doc Chat:** A text-based conversation with a compassionate AI.
        2. **Call Session:** Use your microphone to talk and get a voice-based response.
        3. **Journal & Analysis:** Your conversations are logged and analyzed for insights on your emotional well-being.
    """)
    
    st.divider()
    
    st.header("Daily Mood Tracker")
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            mood_value = st.slider("Rate your mood today:", 1, 10, 5)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Log Mood", key="log_mood_button"):
                st.session_state.mood_history.append({'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'mood': mood_value})
                st.success(f"Mood logged! You rated your mood as {mood_value}/10.")

        if st.session_state.mood_history:
            mood_df = pd.DataFrame(st.session_state.mood_history)
            st.subheader("Your Mood Over Time")
            st.line_chart(mood_df.set_index('date'))

def ai_doc_chat():
    st.title("AI Doc Chat")
    
    with st.container(border=True):
        st.markdown("**Start a conversation with your AI companion.**")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What's on your mind?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_ai_response(st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

def call_session():
    st.title("Call Session")
    st.warning("This is a 'record-and-reply' session, not a live call. Speak when prompted.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", key="start_record"):
            with st.spinner("Recording your message..."):
                audio_file = record_audio(duration=15)
                st.session_state.transcription_text = transcribe_audio(audio_file)
            st.session_state.call_history.append({"speaker": "User", "text": st.session_state.transcription_text, "timestamp": time.time()})
            st.rerun()

    if st.session_state.transcription_text:
        st.write("You said:")
        st.info(st.session_state.transcription_text)

        st.session_state.messages.append({"role": "user", "content": st.session_state.transcription_text})
        
        with st.spinner("AI is thinking and generating a reply..."):
            ai_response = get_ai_response(st.session_state.messages)
            st.session_state.call_history.append({"speaker": "AI", "text": ai_response, "timestamp": time.time()})
            
            with st.chat_message("assistant"):
                st.write(ai_response)
                speak_text(ai_response)
        
        st.session_state.transcription_text = ""
        st.rerun()
    
    with st.expander("Show Call History"):
        for entry in st.session_state.call_history:
            role = "User" if entry['speaker'] == "User" else "AI"
            with st.chat_message(role.lower()):
                st.markdown(f"**{role}:** {entry['text']}")
                
def journal_and_analysis():
    st.title("Journal & Analysis")
    st.markdown("Review your conversations and gain insights into your mood over time.")

    # Journaling Feature
    st.subheader("My Private Journal")
    with st.container(border=True):
        journal_entry = st.text_area("Write down your thoughts:", height=200)
        if st.button("Save Entry"):
            if journal_entry:
                sentiment_score = get_sentiment(journal_entry)['compound']
                st.session_state.daily_journal.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": journal_entry,
                    "sentiment": sentiment_score
                })
                st.success("Journal entry saved!")
            else:
                st.warning("Please write something before saving.")
    
    all_text_entries = " ".join([entry['text'] for entry in st.session_state.call_history if entry['speaker'] == 'User']) + " " + " ".join([entry['text'] for entry in st.session_state.daily_journal])

    st.subheader("Sentiment Analysis")
    sentiment_df, status = analyze_all_sentiment(st.session_state.call_history, st.session_state.daily_journal)

    if status == "Analysis complete.":
        color_map = {
            'Positive': 'green',
            'Neutral': 'yellow',
            'Negative': 'red'
        }
        fig = px.line(sentiment_df, x='date', y='compound', color='sentiment_color', title='Sentiment Over Time', 
                      color_discrete_map=color_map, markers=True)
        st.plotly_chart(fig)

        st.subheader("Key Topics & Words")
        try:
            word_cloud_fig = generate_wordcloud(all_text_entries)
            st.pyplot(word_cloud_fig)
        except NameError:
            st.info("Wordcloud library not installed. Install with `pip install wordcloud`")
    else:
        st.info(status)
    
    with st.expander("Show All History"):
        if st.session_state.daily_journal:
            st.subheader("Journal Entries")
            for entry in st.session_state.daily_journal:
                st.markdown(f"**Date:** {entry['date']} | **Sentiment:** {entry['sentiment']:.2f}")
                st.info(entry['text'])
        
        if st.session_state.call_history:
            st.subheader("Call History")
            for entry in st.session_state.call_history:
                role = "User" if entry['speaker'] == "User" else "AI"
                with st.chat_message(role.lower()):
                    st.markdown(f"**{role}:** {entry['text']}")

def personalized_report():
    st.title("Your Personalized Wellness Report")
    st.markdown("A complete summary of your emotional trends and insights.")
    
    all_text_entries = " ".join([entry['text'] for entry in st.session_state.call_history if entry['speaker'] == 'User']) + " " + " ".join([entry['text'] for entry in st.session_state.daily_journal])
    sentiment_df, status = analyze_all_sentiment(st.session_state.call_history, st.session_state.daily_journal)
    
    if status != "Analysis complete.":
        st.warning("Please interact with the app first to generate data for the report.")
        return

    # --- Insight Section ---
    st.header("1. Emotional Trends")
    total_entries = len(sentiment_df)
    positive_count = len(sentiment_df[sentiment_df['sentiment_color'] == 'Positive'])
    negative_count = len(sentiment_df[sentiment_df['sentiment_color'] == 'Negative'])
    
    st.markdown(f"This report is based on **{total_entries}** interactions you've had with the app. Let's take a closer look at your emotional trends.")
    
    if positive_count > negative_count:
        st.success("Your mood trends are generally positive! Keep focusing on the good things.")
    elif negative_count > positive_count:
        st.warning("It looks like you've been going through some difficult moments. Remember to be kind to yourself.")
        st.info("The AI companion is here to listen, and it's also a great idea to consider reaching out to a professional.")
    else:
        st.info("Your sentiment has been quite balanced. Keep monitoring your mood to stay in tune with your feelings.")

    # --- Charts & Visualization ---
    st.header("2. Detailed Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Over Time")
        color_map = {'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'}
        fig = px.line(sentiment_df, x='date', y='compound', color='sentiment_color', markers=True, 
                      color_discrete_map=color_map, title='Your Emotional Journey')
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Key Topics")
        try:
            word_cloud_fig = generate_wordcloud(all_text_entries)
            st.pyplot(word_cloud_fig)
        except NameError:
            st.info("Wordcloud library not installed.")
    
    # --- Download Report ---
    report_text = f"""
    --- Your Personalized Wellness Report ---
    Generated on: {datetime.now().strftime("%B %d, %Y")}

    Emotional Trends Summary:
    Based on your conversations and journal entries, your mood trends are {
        'generally positive' if positive_count > negative_count else
        'leaning towards negative' if negative_count > positive_count else
        'quite balanced'
    }.

    Total Entries Analyzed: {total_entries}
    - Positive Interactions: {positive_count}
    - Negative Interactions: {negative_count}
    - Neutral Interactions: {total_entries - positive_count - negative_count}

    Journal & Chat Content:
    {all_text_entries}

    --- End of Report ---
    """
    
    st.download_button(
        label="Download Report as Text File",
        data=report_text,
        file_name="wellness_report.txt",
        mime="text/plain",
    )

# --- Navigation logic ---
page = st.sidebar.radio("Go to:", ('Home', 'AI Doc Chat', 'Call Session', 'Journal & Analysis', 'Personalized Report'))

if page == 'Home':
    st.session_state['page'] = 'home'
    homepage()
elif page == 'AI Doc Chat':
    st.session_state['page'] = 'chat'
    ai_doc_chat()
elif page == 'Call Session':
    st.session_state['page'] = 'call'
    call_session()
elif page == 'Journal & Analysis':
    st.session_state['page'] = 'journal'
    journal_and_analysis()
elif page == 'Personalized Report':
    st.session_state['page'] = 'report'
    personalized_report()