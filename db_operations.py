# db_operations.py
from supabase import create_client
import streamlit as st
import pandas as pd
from datetime import datetime

@st.cache_resource(show_spinner=False)
def setup_supabase_client(url: str, key: str):
    """Initializes the Supabase client."""
    if not url or not key:
        return None, False
    try:
        client = create_client(url, key)
        return client, True
    except Exception as e:
        st.error(f"DB connection failed: {e}") # Crucial UX improvement
        return None, False

def register_user_db(supabase_client, email: str):
    """Registers a new user."""
    try:
        res = supabase_client.table("users").insert({"email": email}).execute()
        if getattr(res, "data", None):
            return res.data[0].get("id")
    except Exception as e:
        st.error(f"Database registration failed: {e}")
        return None

def get_user_by_email_db(supabase_client, email: str):
    """Retrieves a user by email."""
    try:
        res = supabase_client.table("users").select("*").eq("email", email).execute()
        return res.data or []
    except Exception:
        return []

# --- SAVE Operations ---

def save_journal_db(supabase_client, user_id, text: str, sentiment: float) -> bool:
    try:
        supabase_client.table("journal_entries").insert({"user_id": user_id, "entry_text": text, "sentiment_score": float(sentiment)}).execute()
        return True
    except Exception as e:
        st.error(f"Journal save failed: {e}")
        return False

def save_mood_db(supabase_client, user_id, mood: int, note: str) -> bool:
    try:
        supabase_client.table("mood_logs").insert({"user_id": user_id, "mood_score": mood, "note": note}).execute()
        return True
    except Exception as e:
        st.error(f"Mood log save failed: {e}")
        return False

def save_phq9_db(supabase_client, user_id, score: int, interpretation: str) -> bool:
    try:
        supabase_client.table("phq9_scores").insert({"user_id": user_id, "score": score, "interpretation": interpretation}).execute()
        return True
    except Exception as e:
        st.error(f"PHQ-9 save failed: {e}")
        return False
        
def save_ece_log_db(supabase_client, user_id, filtered_hr: float, gsr_stress: float, mood_score: int) -> bool:
    try:
        supabase_client.table("ece_logs").insert({
            "user_id": user_id, 
            "filtered_hr": filtered_hr, 
            "gsr_stress": gsr_stress,
            "mood_score": mood_score
        }).execute()
        return True
    except Exception as e:
        st.error(f"ECE log save failed: {e}")
        return False

# --- LOAD Operation ---

@st.cache_data(show_spinner=False)
def load_all_user_data(user_id, supabase_client):
    """Loads all data types for the user."""
    if not supabase_client:
        return {"journal": [], "mood": [], "phq9": [], "ece": []}
    
    data = {}
    try:
        # Load Journal, Mood, PHQ-9, ECE... (Loading logic remains the same)
        # ... (implementation is long, but the structure remains the same) ...
        
        # Load Journal
        res_j = supabase_client.table("journal_entries").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["journal"] = [{"date": e.get("created_at"), "text": e.get("entry_text"), "sentiment": e.get("sentiment_score")} for e in res_j.data or []]
        
        # Load Mood
        res_m = supabase_client.table("mood_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["mood"] = [{"date": e.get("created_at"), "mood": e.get("mood_score"), "note": e.get("note")} for e in res_m.data or []]
        
        # Load PHQ-9 (just the latest one)
        res_p = supabase_client.table("phq9_scores").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        data["phq9"] = res_p.data or []

        # Load ECE History
        res_e = supabase_client.table("ece_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        data["ece"] = [{"date": e.get("created_at"), "hr": e.get("filtered_hr"), "stress": e.get("gsr_stress"), "mood": e.get("mood_score")} for e in res_e.data or []]
        
    except Exception as e:
        st.error(f"Data load failed: {e}")
        # Fallback to empty lists on failure
        return {"journal": [], "mood": [], "phq9": [], "ece": []}
        
    return data