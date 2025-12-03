Youth Wellness – AI-Powered Mental Health & Wellness Companion
Youth Wellness is a holistic, AI-enhanced mental well-being platform that helps users track their emotions, reflect through CBT-based exercises, access mindfulness tools, and engage with a supportive AI companion.
The app securely stores user data using Supabase, ensuring privacy and reliability.
Built using Streamlit, Supabase, OpenRouter LLMs, and Python.

🌟 Features

1. Home Dashboard
A clean and welcoming overview of all features.
2. Mindful Journaling
Users can write and save journal entries.
Entries are securely stored in Supabase with timestamps.
AI prompts help deepen reflection
3. Mindful Journaling
Users can write and save journal entries.
Entries are securely stored in Supabase with timestamps.
AI prompts help deepen reflection.
4. Mood Tracker
Users log their mood daily.
Stored in Supabase.
Visualized with charts to show emotional patterns.
5. CBT Thought Record
CBT-based structured reflection, including:
Situation
Automatic Thoughts
Cognitive Distortions
Balanced Thoughts
Outcome/Reflection
Records are saved in Supabase for personal tracking.
6. AI Chat (OpenRouter)
An AI companion that helps with:
Emotional reflection
Motivation
Cognitive reframing
Journaling prompts
7. Wellness Check-in (PHQ-9 Assessment)
A clinically validated PHQ-9 self-assessment.
No results are stored unless the user chooses to save mood reflections.
8. Mindful Breathing
A guided breathing module for stress relief.
9. IoT Dashboard (ECE Feature)
Optional page that fetches relevant sensor data (if connected) via Supabase or local files.
10. Report & Summary
Mood history
Journal highlights
CBT summaries
Weekly or monthly patterns

🛠️ Tech Stack
Python
Streamlit
Supabase (PostgreSQL + Auth)
OpenRouter API
Pandas, Plotly
Streamlit Components

🧩 Supabase Setup
1. Create a Supabase project
Go to: https://supabase.com/
2. Create tables
You likely used tables such as:
journals
moods
cbt_records
users
iot_data
Each contains columns like:
id
user_id
timestamp
content / mood_value / thoughts, etc.
3. Add your API keys in Streamlit
Create this file:
/.streamlit/secrets.toml
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_anon_key"
OPENROUTER_API_KEY = "your_openrouter_key"
4. Install Supabase Python client
pip install supabase
5. Initialize in your app
from supabase import create_client

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)
🚀 Run the Project Locally
git clone https://github.com/your-username/Youth-Wellness.git
cd Youth-Wellness
pip install -r requirements.txt
streamlit run app.py

🔐 Privacy Notice
User data is stored securely in Supabase under their own account.
No PHQ-9 answers are stored unless added as reflections.
AI conversations are not saved.

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

🤝 Contributions
Pull requests are welcome

⭐ Acknowledgements
Supabase
Streamlit
OpenRouter
PHQ-9 Public Domain Questionnaire
