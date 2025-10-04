import streamlit as st
from supabase import create_client, Client
import os
from datetime import datetime

# --- CONFIGURATION (UPDATE WITH YOUR ACTUAL KEYS) ---
# IMPORTANT: Replace the placeholder values below with your actual Supabase URL and Key.
# If you are using Streamlit Cloud, it's safer to use st.secrets instead of os.environ.get
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL_HERE")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY_HERE")

# --- CUSTOM CSS FOR STYLING (Matching your screenshot) ---

def apply_custom_styles():
    """Applies custom CSS for card styling and overall layout."""
    st.markdown("""
    <style>
    /* Main Streamlit container adjustments for wider content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* General card styling for metric boxes/containers to add shadows and rounded corners */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMetric"]) {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        padding: 10px;
        margin-bottom: 15px;
        background-color: white;
    }
    
    /* Custom style for the 'Overview' and 'Home' links/buttons in the sidebar */
    div[data-testid="stSidebarUserContent"] button {
        background-color: #f0f0f0; 
        color: #1E90FF; /* Dodger Blue for links */
        border-radius: 8px;
        border: none;
        margin: 5px 0;
        width: 100%;
        text-align: left;
        padding: 10px 15px;
        font-weight: bold;
    }

    div[data-testid="stSidebarUserContent"] button:hover {
        background-color: #E6E6FA; /* Light Lavender on hover */
    }

    </style>
    """, unsafe_allow_html=True)

# --- SUPABASE UTILITIES ---

@st.cache_resource
def init_connection() -> Client:
    """Initializes and caches the Supabase connection."""
    # Ensure keys are set before attempting to create a client
    if SUPABASE_URL == "YOUR_SUPABASE_URL_HERE" or SUPABASE_KEY == "YOUR_SUPABASE_ANON_KEY_HERE":
        st.error("Please update the SUPABASE_URL and SUPABASE_KEY in app.py with your actual credentials.")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ⭐️ FIX: Use _supabase_client to avoid UnhashableParamError
@st.cache_data(show_spinner="Loading user data...")
def load_all_user_data(user_id, _supabase_client: Client):
    """
    Loads all profile data for a given user ID.
    The '_supabase_client' parameter is ignored by st.cache_data for hashing.
    """
    if not user_id or not _supabase_client:
        return None

    response = _supabase_client.table("profiles").select("*").eq("id", user_id).execute()
    
    # Assuming 'profiles' table has a row for the user
    return response.data[0] if response.data else None

# --- AUTHENTICATION LOGIC ---

def sign_in(email, password, supabase_client):
    """Handles user sign-in."""
    if not supabase_client:
        st.error("Supabase client is not initialized.")
        return
        
    try:
        response = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
        
        if response.user and response.user.id:
            st.session_state["user_id"] = response.user.id
            st.session_state["user_email"] = response.user.email
            st.session_state["is_authenticated"] = True
            # Set the initial page upon successful login
            st.session_state["current_page"] = "home" 
            st.rerun()
        else:
            st.error("Sign-in failed. Please check your credentials.")
    except Exception as e:
        # Catch specific API errors if possible
        st.error(f"An error occurred during sign-in. Details: {e}")

def sign_out():
    """Handles user sign-out."""
    if st.session_state.get("_supabase_client_obj"):
        try:
            st.session_state["_supabase_client_obj"].auth.sign_out()
        except Exception:
             # Ignore errors on sign out if the connection is already dropped
             pass 

    # Clear session state variables
    for key in ["user_id", "user_email", "is_authenticated", "current_page", "user_data"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- LAYOUT AND PAGE FUNCTIONS ---

def sidebar_auth():
    """Renders the sidebar with login/logout and navigation."""
    st.sidebar.title("Youth Wellness App")
    
    if st.session_state.get("is_authenticated"):
        
        # ⭐️ Load data using the fixed function call
        user_data = load_all_user_data(
            st.session_state["user_id"],
            _supabase_client=st.session_state.get("_supabase_client_obj")
        )
        st.session_state["user_data"] = user_data # Store loaded data

        st.sidebar.write(f"Logged in as: **{st.session_state.get('user_email', 'User')}**")
        
        st.sidebar.markdown("---")
        st.sidebar.header("Navigation")
        
        # Navigation Links/Buttons
        if st.sidebar.button("🏠 Home", key="nav_home"):
            st.session_state["current_page"] = "home"
            
        if st.sidebar.button("Overview", key="nav_overview"):
            st.session_state["current_page"] = "overview"

        st.sidebar.markdown("---")
        st.sidebar.button("Logout", on_click=sign_out, key="logout_btn")

    else:
        st.sidebar.header("Login")
        with st.sidebar.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In")
            
            if submitted:
                supabase_client = st.session_state.get("_supabase_client_obj")
                sign_in(email, password, supabase_client)

def home_page():
    """Renders the main home page content (dashboard view)."""
    st.title("Welcome to the Youth Wellness Dashboard")
    st.subheader("Quick Insights")

    # Example of styled "cards" using st.columns and st.metric
    col1, col2, col3 = st.columns(3)

    # These metrics simulate the cards in your screenshot
    with col1:
        st.metric(label="Total Participants", value="1,245", delta="12%")
        
    with col2:
        st.metric(label="Average Engagement Score", value="4.7/5", delta="0.2")

    with col3:
        st.metric(label="Pending Tasks", value="8", delta="-2")
        
    st.markdown("---")
    
    st.header("Latest Updates")
    st.info("The application has successfully fixed the caching issue and is ready.")
    st.write("Use the sidebar navigation to move between sections.")

def overview_page():
    """Renders the Overview page."""
    st.title("Overview and Detailed Statistics")
    st.write(f"Hello, {st.session_state.get('user_email', 'User')}! Here is your detailed data overview.")
    
    # Placeholder chart
    st.line_chart({"data": [10, 20, 15, 30, 25, 40]})
    
# --- MAIN APPLICATION LOGIC ---

if __name__ == "__main__":
    # 1. Initialize session state and connections
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "home" # Default page

    # Initialize and store the Supabase client object in session_state
    if "_supabase_client_obj" not in st.session_state:
        st.session_state["_supabase_client_obj"] = init_connection()

    # 2. Apply Custom Styling
    apply_custom_styles()

    # 3. Render Sidebar and handle authentication/navigation
    sidebar_auth()

    # 4. Render Main Content based on authentication status and current page
    if st.session_state["is_authenticated"]:
        if st.session_state["current_page"] == "home":
            home_page()
        elif st.session_state["current_page"] == "overview":
            overview_page()
    else:
        # Content displayed when not logged in
        st.header("Welcome!")
        st.info("Please sign in using the form in the sidebar to access the dashboard.")
        # You can choose to show a truncated version of the home page or a marketing page here
        # home_page()