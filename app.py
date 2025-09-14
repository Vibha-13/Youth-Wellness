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
from supabase import create_client, Client

# --- Supabase Initialization ---
# Get secrets from Streamlit Cloud
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

# Create a Supabase client
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.sidebar.success("Database Connected 🟢")
else:
    st.sidebar.warning("Database not connected 🔴")

# --- Custom Styling & Theme ---
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
    color: #1c1c1c;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 15px;
}

/* Chat Bubbles */
.stChatMessage.user {
    background: #4facfe;
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 18px;
    margin: 5px;
}
.stChatMessage.assistant {
    background: #43e97b;
    color: white;
    border-radius: 18px 18px 18px 0;
    padding: 12px 18px;
    margin: 5px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #ff758c, #ff7eb3);
    color: white;
    border-radius: 30px;
    padding: 8px 20px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Navigation")

# --- Load API Key and Configure Model ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model = genai.GenerativeModel('gemini-pro')
        st.sidebar.markdown("Status: **AI API Connected** 🟢")
    except Exception as e:
        st.sidebar.markdown("Status: **AI API Failed** 🔴")
        st.error(f"API Configuration Error: {e}")
else:
    st.sidebar.markdown("Status: **Local Demo Mode** 🟠")
    st.warning("No API key found. The app is running in local demo mode. To enable the AI API, add your key to a Streamlit Cloud Secret.")

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
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None

# --- Core Functions ---
def get_ai_response(prompt_messages):
    """
    Attempts to get a response from an AI API,
    falling back to a local response if the API call fails.
    """
    analyzer = SentimentIntensityAnalyzer()
    last_user_message = prompt_messages[-1]['content']
    sentiment_score = analyzer.polarity_scores(last_user_message)['compound']

    if 'model' in globals():
        try:
            response = model.generate_content(last_user_message)
            return response.text
        except Exception as e:
            st.warning("AI API failed. Using local model for now.")
            st.error(f"Full API Error: {e}")
            # Fallback will run below
    else:
        # Fallback will run below
        pass

    # --- Local Fallback Logic (if AI fails or is not configured) ---
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

# --- Supabase Database Functions ---
def get_user_by_email(email):
    response = supabase.table('users').select('*').eq('email', email).execute()
    return response.data

def register_user(email):
    response = supabase.table('users').insert({"email": email}).execute()
    if response.data:
        st.session_state['user_id'] = response.data[0]['id']
        st.session_state['logged_in'] = True
        st.session_state['user_email'] = email
        st.success("Registration successful! You are now logged in.")
        st.rerun()

def save_journal_entry_to_db(entry_text, sentiment_score):
    if st.session_state.logged_in and st.session_state.user_id:
        data = {
            "user_id": st.session_state.user_id,
            "entry_text": entry_text,
            "sentiment_score": float(sentiment_score)
        }
        supabase.table('journal_entries').insert(data).execute()

def load_journal_entries_from_db():
    if st.session_state.logged_in and st.session_state.user_id:
        response = supabase.table('journal_entries').select('*').eq('user_id', st.session_state.user_id).order('created_at').execute()
        st.session_state.daily_journal = [
            {'date': entry['created_at'], 'text': entry['entry_text'], 'sentiment': entry['sentiment_score']}
            for entry in response.data
        ]

def user_authentication():
    st.sidebar.subheader("User Authentication")
    if not st.session_state.logged_in:
        email = st.sidebar.text_input("Enter your email:")
        if st.sidebar.button("Login / Register"):
            if email:
                user = get_user_by_email(email)
                if user:
                    st.session_state['user_id'] = user[0]['id']
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = email
                    st.sidebar.success("Login successful!")
                    load_journal_entries_from_db()
                    st.rerun()
                else:
                    register_user(email)
            else:
                st.sidebar.warning("Please enter a valid email.")
    else:
        st.sidebar.write("Logged in as:", st.session_state.user_email)
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['user_id'] = None
            st.session_state['user_email'] = None
            st.session_state['daily_journal'] = []
            st.session_state['messages'] = []
            st.sidebar.info("Logged out.")
            st.rerun()

def mindful_breathing():
    st.title("Find Your Calm")
    st.markdown("Let your breath guide you back to peace and center.")

    if 'breathing_phase' not in st.session_state:
        st.session_state.breathing_phase = "Inhale"
    if 'current_cycle' not in st.session_state:
        st.session_state.current_cycle = 0
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    breathing_pattern = {
        "Inhale": 4,
        "Hold (top)": 4,
        "Exhale": 6,
    }
    
    phases = list(breathing_pattern.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start", disabled=st.session_state.is_running)
    with col2:
        reset_button = st.button("Reset")

    if reset_button:
        st.session_state.breathing_phase = "Inhale"
        st.session_state.current_cycle = 0
        st.session_state.is_running = False
        st.rerun()

    if start_button:
        st.session_state.is_running = True
        st.rerun()

    if st.session_state.is_running:
        for cycle in range(st.session_state.current_cycle, 5):
            for phase in phases:
                st.session_state.breathing_phase = phase
                st.session_state.current_cycle = cycle + 1
                
                with st.expander("4-4-6 Breathing Exercise"):
                    st.markdown(f"**Current Phase: {phase}**")
                    with st.empty():
                        for t in range(breathing_pattern[phase], 0, -1):
                            st.subheader(f"Breathe... {t}")
                            time.sleep(1)
                
            st.write(f"Cycle {st.session_state.current_cycle} of 5 completed.")
            if st.session_state.current_cycle == 5:
                st.balloons()
                st.session_state.is_running = False
                st.experimental_rerun()

    st.subheader("Progress")
    overall_progress = (st.session_state.current_cycle) / 5
    st.progress(overall_progress)
    st.write(f"Overall Progress: {int(overall_progress * 100)}%")

def mindful_journaling():
    st.title("Mindful Journaling")
    
    with st.container(border=True):
        st.markdown("<div class='card-lavender'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <p style='font-weight: bold;'>Today's Reflection:</p>
            <p>What made you smile today, even if it was small?</p>
            """, unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    journal_entry = st.text_area("Pour your heart out here... There's no judgment, only compassion. Write whatever feels right for you in this moment.", height=200)
    
    if st.button("Save Entry", key="journal_save_button"):
        if journal_entry:
            sentiment_score = get_sentiment(journal_entry)['compound']
            if st.session_state.logged_in:
                save_journal_entry_to_db(journal_entry, sentiment_score)
            else:
                st.session_state.daily_journal.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "text": journal_entry,
                    "sentiment": sentiment_score
                })
            st.success("Journal entry saved successfully!")
            st.rerun()
        else:
            st.warning("Please write something before saving.")

# --- Main app pages ---
def homepage():
    st.title("Your Wellness Sanctuary")
    st.markdown("A safe space designed with therapeutic colors and gentle interactions to support your mental wellness journey.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Daily Inspiration")
        with st.container(border=True):
            st.markdown("<div class='quote-box'>", unsafe_allow_html=True)
            st.markdown("<h3>Words of Hope</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <p style='font-style: italic; font-size: 1.2rem; margin-top: 20px;'>
                    "You are braver than you believe, stronger than you seem, and more loved than you know."
                </p>
                <p style='text-align: right; margin-top: 10px;'>
                    - A.A. Milne
                </p>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1549490349-f06b3e942007?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Take a moment for yourself")

    st.header("How It Works")
    with st.container(border=True):
        st.markdown("""
            1. **AI Doc Chat:** Have a text-based conversation with a compassionate AI.
            2. **Call Session:** Use your microphone to talk and get a voice-based response.
            3. **Journal & Analysis:** Your conversations are logged and analyzed for insights on your emotional well-being.
        """)

    st.divider()

    st.header("Daily Mood Tracker")
    with st.container(border=True):
        col3, col4 = st.columns([2, 1])
        with col3:
            mood_value = st.slider("Rate your mood today:", 1, 10, 5)
        with col4:
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

    st.subheader("My Private Journal")
    with st.container(border=True):
        journal_entry = st.text_area("Write down your thoughts:", height=200)
        if st.button("Save Entry"):
            if journal_entry:
                sentiment_score = get_sentiment(journal_entry)['compound']
                if st.session_state.logged_in:
                    save_journal_entry_to_db(journal_entry, sentiment_score)
                else:
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
    
def emotional_journey():
    st.title("Your Emotional Journey")
    st.markdown("A narrative generated by the AI based on your conversations and journal entries.")
    
    all_text_entries = " ".join([entry['text'] for entry in st.session_state.call_history if entry['speaker'] == 'User']) + " " + " ".join([entry['text'] for entry in st.session_state.daily_journal])
    
    if not all_text_entries:
        st.warning("Please interact with the app first to generate data for your emotional journey.")
        return

    sentiment_df, status = analyze_all_sentiment(st.session_state.call_history, st.session_state.daily_journal)
    
    if status == "Analysis complete.":
        avg_sentiment = sentiment_df['compound'].mean()
        if avg_sentiment > 0.1:
            st.image("https://images.unsplash.com/photo-1510525008061-f09c259d6820?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
            st.success("Your journey has been filled with positive energy. ✨")
        elif avg_sentiment < -0.1:
            st.image("https://images.unsplash.com/photo-1481026469466-2619c991b103?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
            st.warning("Your journey has been challenging, but you are strong. 💧")
        else:
            st.image("https://images.unsplash.com/photo-1508247966967-b52b212f7194?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
            st.info("Your journey has been a mix of ups and downs, full of growth. 🌱")

    st.subheader("The Story So Far...")
    with st.spinner("The AI is writing your story..."):
        prompt = f"""
        Based on the following journal entries and conversations, write a personalized, empathetic, and uplifting short story about the user's emotional journey. Use a narrative style, focusing on their growth and resilience. Do not mention specific names or events.

        User data:
        "{all_text_entries}"

        Write the story from the perspective of an encouraging observer. The story should be 3-4 paragraphs long.
        """
        
        try:
            story = model.generate_content(prompt).text
            st.markdown(story)
        except Exception:
            st.error("I'm sorry, I couldn't generate the story right now. Please try again later.")

def crisis_support():
    st.title("Crisis & Immediate Support")
    st.markdown("You are not alone. If you're going through a tough time, help is available. These feelings are temporary, and reaching out shows incredible strength.")
    
    st.header("Crisis Hotlines")
    st.info("Reach out to these numbers if you are in immediate crisis or need to speak with someone.")
    
    st.markdown("### National Suicide & Crisis Lifeline")
    st.markdown("#### **Call or text 988**")
    st.markdown("24/7 free and confidential support.")
    
    st.markdown("### SAMHSA National Helpline")
    st.markdown("#### **1-800-662-4357**")
    st.markdown("Treatment referral and information service.")
    
    st.markdown("### Crisis Text Line")
    st.markdown("#### **Text HOME to 741741**")
    st.markdown("24/7 crisis support via text message.")
    
    st.divider()
    
    st.header("Quick Grounding Exercise")
    st.info("Try the 5-4-3-2-1 technique to ground yourself in the present moment.")
    
    st.markdown("1. **5 Things** you can **see**.")
    st.markdown("2. **4 Things** you can **touch**.")
    st.markdown("3. **3 Things** you can **hear**.")
    st.markdown("4. **2 Things** you can **smell**.")
    st.markdown("5. **1 Thing** you can **taste**.")

# --- Navigation logic ---
user_authentication()

page = st.sidebar.radio("Go to:", ('Home', 'AI Doc Chat', 'Call Session', 'Mindful Journaling', 'Journal & Analysis', 'Personalized Report', 'My Emotional Journey', 'Mindful Breathing', 'Crisis Support'))

if page == 'Home':
    st.session_state['page'] = 'home'
    homepage()
elif page == 'AI Doc Chat':
    st.session_state['page'] = 'chat'
    ai_doc_chat()
elif page == 'Call Session':
    st.session_state['page'] = 'call'
    call_session()
elif page == 'Mindful Journaling':
    st.session_state['page'] = 'journaling'
    mindful_journaling()
elif page == 'Journal & Analysis':
    st.session_state['page'] = 'journal'
    journal_and_analysis()
elif page == 'Personalized Report':
    st.session_state['page'] = 'report'
    personalized_report()
elif page == 'My Emotional Journey':
    st.session_state['page'] = 'journey'
    emotional_journey()
elif page == 'Mindful Breathing':
    st.session_state['page'] = 'breathing'
    mindful_breathing()
elif page == 'Crisis Support':
    st.session_state['page'] = 'crisis'
    crisis_support()