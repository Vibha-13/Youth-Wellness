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

# --- App configuration and state management ---
st.set_page_config(
    page_title="AI Wellness Companion",
    page_icon="🧠",
    layout="wide",
)

st.sidebar.header("Navigation")

# Use st.session_state to manage pages and data
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

# --- Local Fallback Functions ---
def get_ai_response(prompt_messages):
    """Provides a smarter, local-only response based on user input for the demo."""
    last_user_message = prompt_messages[-1]['content'].lower() if prompt_messages else ""

    if "sad" in last_user_message or "depressed" in last_user_message:
        return "I hear the heaviness in your words. It's okay to feel this way. What is one small thing that could bring you a bit of comfort right now?"
    elif "anxious" in last_user_message or "stressed" in last_user_message or "panic" in last_user_message:
        return "Take a deep breath with me. I'm here. Can you describe what is making you feel this way?"
    elif "happy" in last_user_message or "good" in last_user_message or "great" in last_user_message:
        return "That's wonderful to hear! I'm happy for you. What's one thing you're most grateful for from today?"
    elif "alone" in last_user_message or "lonely" in last_user_message:
        return "You're not alone. I'm here to listen. Would you like to talk about what's been on your mind?"
    else:
        # Default responses for general conversation
        responses = [
            "I hear you. What's on your mind?",
            "That sounds challenging. Can you tell me more?",
            "Thank you for sharing. I'm here to listen.",
            "Your feelings are valid. What happened next?",
        ]
        return random.choice(responses)

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def analyze_call_sentiment(history):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_list = []
    
    if not history:
        return "No conversation to analyze yet."

    for entry in history:
        sentiment = analyzer.polarity_scores(entry['text'])
        sentiment_list.append({
            'speaker': entry['speaker'],
            'text': entry['text'],
            'compound': sentiment['compound'],
            'date': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(entry['timestamp']))
        })

    df = pd.DataFrame(sentiment_list)
    return df

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
    """Provides a slightly more realistic transcription for the demo."""
    dummy_responses = [
        "This is a demo transcription. I'm hearing you clearly. What's on your mind?",
        "This is a demo transcription. It sounds like you have a lot to say. Please continue.",
        "This is a demo transcription. Thank you for speaking. I'm here to listen.",
    ]
    return random.choice(dummy_responses)

def speak_text(text):
    """Uses a local TTS engine to speak text."""
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
        1. **AI Doc Chat:** Have a text-based conversation with a compassionate AI.
        2. **Call Session:** Use your microphone to talk and get a voice-based response.
        3. **Journal & Analysis:** Your conversations are logged and analyzed for insights on your emotional well-being.
    """)
    st.sidebar.markdown("Status: **Demo Mode**")

def ai_doc_chat():
    st.title("AI Doc Chat")
    
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
    
    st.header("Call History")
    for entry in st.session_state.call_history:
        role = "User" if entry['speaker'] == "User" else "AI"
        with st.chat_message(role.lower()):
            st.markdown(f"**{role}:** {entry['text']}")

def journal_and_analysis():
    st.title("Journal & Analysis")
    st.markdown("Review your conversations and gain insights into your mood over time.")

    all_text = " ".join([entry['text'] for entry in st.session_state.call_history if entry['speaker'] == 'User'])
    
    if st.button("Analyze My Journal"):
        st.session_state.analysis_text = all_text
        st.rerun()

    if st.session_state.analysis_text:
        st.subheader("Sentiment Analysis")
        sentiment_df = analyze_call_sentiment(st.session_state.call_history)
        if isinstance(sentiment_df, pd.DataFrame) and not sentiment_df.empty:
            st.dataframe(sentiment_df[['speaker', 'text', 'compound']])
            fig = px.line(sentiment_df, x='date', y='compound', color='speaker', title='Sentiment Over Time')
            st.plotly_chart(fig)
            st.subheader("Key Topics & Words")
            try:
                word_cloud_fig = generate_wordcloud(st.session_state.analysis_text)
                st.pyplot(word_cloud_fig)
            except NameError:
                st.info("Wordcloud library not installed. Install with `pip install wordcloud`")
        else:
            st.info("No conversations to analyze yet. Start a chat or a call session!")
    
    st.header("Full Journal History")
    for entry in st.session_state.call_history:
        role = "User" if entry['speaker'] == "User" else "AI"
        with st.chat_message(role.lower()):
            st.markdown(f"**{role}:** {entry['text']}")

# --- Navigation logic ---
page = st.sidebar.radio("Go to:", ('Home', 'AI Doc Chat', 'Call Session', 'Journal & Analysis'))

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