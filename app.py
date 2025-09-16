# Standard library imports
import os
import io
import time
import base64
import requests

# Third-party imports for data handling and visualization
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Third-party imports for Google Gemini API and audio
import google.generativeai as genai
import wavio
import sounddevice as sd

# Firebase and Supabase imports
import supabase
from supabase import create_client, Client
from gotrue import AuthResponse

# Configuration for Firebase and Supabase, including environment variables
st.set_page_config(layout="wide", page_title="Gemini AI App")
st.title("Gemini AI App")

# Function to safely retrieve secrets
def get_secret(key):
    """
    Retrieves a secret from Streamlit secrets, falling back to environment variables.
    """
    return st.secrets.get(key) or os.getenv(key)

# API keys and Supabase credentials
try:
    gemini_api_key = get_secret("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API key not found. Please set `GEMINI_API_KEY` in Streamlit secrets or as an environment variable.")
        st.stop()
    genai.configure(api_key=gemini_api_key)

    supabase_url = get_secret("SUPABASE_URL")
    supabase_key = get_secret("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        st.error("Supabase URL or key not found. Please set `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets.")
        st.stop()
    
    supabase_client: Client = create_client(supabase_url, supabase_key)
    
except Exception as e:
    st.error(f"Error during API configuration: {e}")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# Sidebar for user authentication and settings
with st.sidebar:
    st.header("Authentication")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In"):
            if email and password:
                try:
                    auth_response: AuthResponse = supabase_client.auth.sign_in_with_password(
                        {"email": email, "password": password}
                    )
                    st.session_state.is_authenticated = True
                    st.session_state.user = auth_response.user
                    st.success("Signed in successfully!")
                    
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")
            else:
                st.warning("Please enter both email and password.")

    with col2:
        if st.button("Sign Up"):
            if email and password:
                try:
                    auth_response: AuthResponse = supabase_client.auth.sign_up(
                        {"email": email, "password": password}
                    )
                    st.session_state.is_authenticated = True
                    st.session_state.user = auth_response.user
                    st.success("Signed up and signed in!")
                except Exception as e:
                    st.error(f"Sign-up failed: {e}")
            else:
        st.warning("Please enter both email and password.")
    
    if st.session_state.is_authenticated:
        st.write(f"Logged in as: {st.session_state.user.email}")
        if st.button("Sign Out"):
            supabase_client.auth.sign_out()
            st.session_state.is_authenticated = False
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.user = None
            st.success("Signed out.")

    st.header("Settings")
    model_name = st.selectbox(
        "Select a Model",
        ("gemini-1.5-pro", "gemini-1.0-pro", "gemini-1.5-flash")
    )

    if st.session_state.is_authenticated:
        if st.button("Generate Report"):
            try:
                # Assuming chat history is stored as JSON in Supabase
                response = supabase_client.from_("chat_sessions").select("*").eq("user_id", st.session_state.user.id).execute()
                
                if response.data:
                    chat_data = response.data[0].get("history")
                    
                    st.header("Chat Report")
                    df = pd.DataFrame(chat_data)
                    st.dataframe(df)

                    # Sentiment Analysis
                    analyzer = SentimentIntensityAnalyzer()
                    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
                    fig = px.bar(df, x=df.index, y='sentiment', title='Sentiment Over Time')
                    st.plotly_chart(fig)

                    # Word Cloud
                    all_text = " ".join(df['text'])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                    st.image(wordcloud.to_array(), caption='Word Cloud of Chat', use_column_width=True)

                else:
                    st.warning("No chat history found to generate a report.")
                    
            except Exception as e:
                st.error(f"Error generating report: {e}")

# Function to get a response from the Gemini API
def get_gemini_response(prompt, chat_history):
    model = genai.GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(prompt)
    return response

# Main chat UI
st.title("Gemini Voice & Text Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle audio recording and transcription
def record_audio(duration=5):
    st.info("Recording...")
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    st.success("Recording complete!")
    return recording, fs

def transcribe_audio(audio_data, fs):
    st.info("Transcribing...")
    wav_io = io.BytesIO()
    wavio.write(wav_io, audio_data, fs, sampwidth=2)
    wav_io.seek(0)
    
    audio_base64 = base64.b64encode(wav_io.read()).decode("utf-8")
    
    prompt = "Transcribe the following audio."
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    response = model.generate_content(
        [
            prompt,
            {
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": audio_base64,
                }
            }
        ]
    )
    return response.text

# Buttons for voice chat
voice_col1, voice_col2 = st.columns([1, 6])
with voice_col1:
    if st.button("🎤 Voice Chat"):
        try:
            audio_recording, samplerate = record_audio(duration=5)
            transcribed_text = transcribe_audio(audio_recording, samplerate)
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            
            with st.chat_message("user"):
                st.markdown(transcribed_text)
                
            response = get_gemini_response(transcribed_text, st.session_state.chat_history)
            
            with st.chat_message("assistant"):
                st.markdown(response.text)
                
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.session_state.chat_history.append({"role": "user", "text": transcribed_text})
            st.session_state.chat_history.append({"role": "assistant", "text": response.text})
            
            # Save chat history to Supabase
            if st.session_state.is_authenticated:
                supabase_client.from_("chat_sessions").upsert(
                    {
                        "user_id": st.session_state.user.id, 
                        "history": st.session_state.chat_history
                    }
                ).execute()

        except Exception as e:
            st.error(f"An error occurred during voice chat: {e}")
            
# Handle text input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    try:
        response = get_gemini_response(prompt, st.session_state.chat_history)
        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        st.session_state.chat_history.append({"role": "assistant", "text": response.text})
        
        # Save chat history to Supabase
        if st.session_state.is_authenticated:
            supabase_client.from_("chat_sessions").upsert(
                {
                    "user_id": st.session_state.user.id, 
                    "history": st.session_state.chat_history
                }
            ).execute()
        
    except Exception as e:
        st.error(f"An error occurred during text chat: {e}")