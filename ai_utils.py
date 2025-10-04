# ai_utils.py
import re
import random
from openai import OpenAI, APIError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import time # For the fallback message delay

# AI/LLM Config (Keep these private/env vars in app.py or secrets)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "openai/gpt-3.5-turbo" 

@st.cache_resource
def setup_analyzer():
    """Initializes the VADER Sentiment Analyzer."""
    return SentimentIntensityAnalyzer()

def sentiment_compound(text: str) -> float:
    """Calculates VADER compound sentiment score."""
    if not text:
        return 0.0
    analyzer = setup_analyzer()
    return analyzer.polarity_scores(text)["compound"]

def clean_text_for_ai(text: str) -> str:
    """Cleans text for safe API submission."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def setup_ai_client(api_key: str, history: list):
    """Initializes the AI client and system instruction."""
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

def generate_ai_response(prompt: str, messages: list, ai_client: OpenAI = None, ai_available: bool = False, max_tokens: int = 300):
    """
    Handles AI generation with safety catches and API error handling.
    """
    prompt_lower = prompt.lower()
    
    # 1. CRISIS SAFETY CATCH (Local & Immediate)
    if any(phrase in prompt_lower for phrase in ["hurt myself", "end it all", "suicide", "better off dead", "kill myself"]):
        return (
            "**🛑 STOP. This is an emergency.** Please contact help immediately. Your safety is the most important thing. **Call or text 988 (US/Canada) or a local crisis line NOW.** You can also reach out to a trusted family member or teacher. Hold on, you are not alone. Let's try the 5-4-3-2-1 grounding technique together: Name 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste."
        )

    # 2. Sentimental/Canned Response (If AI is not available)
    if not ai_available or not ai_client:
        canned = [
            "Thanks for sharing. I hear you — would you like to tell me more?",
            "That’s a lot to carry. I’m here. Could you describe one small thing that feels heavy right now?",
            "I’m listening. If you want, we can try a 1-minute breathing exercise together."
        ]
        time.sleep(1) # Simulate thinking time
        return random.choice(canned)
    
    # 3. Default AI generation
    prompt_clean = clean_text_for_ai(prompt)

    # Append new user message before sending to API
    if messages[-1]["content"] != prompt_clean or messages[-1]["role"] != "user":
         messages.append({"role": "user", "content": prompt_clean})

    try:
        # Use system message + last 10 messages for context
        context_messages = messages[-11:]
        
        resp = ai_client.chat.completions.create(
            model=OPENROUTER_MODEL_NAME,
            messages=context_messages,
            max_tokens=max_tokens,
            temperature=0.7 
        )
        
        if resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content
        
    except APIError as e:
         st.error(f"AI API Error: {e.body.get('message', 'Unknown error')}")
    except Exception as e:
         st.error(f"An unexpected error occurred during AI generation: {e}")
         
    # Fallback if API fails
    return "I’m sorry, I'm having trouble connecting right now. Let's try a simple coping exercise instead, like 3 deep breaths. 🧘‍♀️"