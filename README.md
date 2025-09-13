# 🧠 AI Wellness Companion

## 💡 The Problem We're Solving

Mental and emotional wellness can feel intimidating and isolating, especially for young people. Accessing professional help is often a hurdle, and many people need a safe, private space to express their thoughts and feelings without judgment. We created the AI Wellness Companion to provide a discreet, accessible, and supportive tool for this exact purpose.

## 🚀 Key Features

* **AI Doc Chat:** A text-based conversational chatbot that offers a compassionate and listening ear.
* **Call Session:** A "record-and-reply" voice session that uses **Speech-to-Text (STT)** to transcribe the user's voice and **Text-to-Speech (TTS)** to generate a voice-based response.
* **Journal & Analysis:** All conversations are logged, providing a journal that can be analyzed for insights into your emotional well-being.

## ⚙️ Our Core Technology

This application is built with **Python** and **Streamlit** for a quick and powerful UI. Our core design philosophy is **resilience**.

-   **Graceful Degradation:** While the full version integrates advanced APIs (like Whisper for STT and ElevenLabs for TTS), we have built a local-first system with robust fallbacks. This ensures the app is always functional and provides a reliable user experience, even if external services are unavailable.
-   **Analytics:** We use `vaderSentiment` to perform sentiment analysis and `matplotlib`/`plotly` to visualize the data, giving users insights into their mood.

## 🛣️ Our Future Roadmap

For this hackathon, we focused on building a Minimum Viable Product (MVP) that is guaranteed to be stable and reliable. For our next steps, we plan to:

* **Integrate APIs:** Re-enable the full suite of API integrations for enhanced conversation and voice quality.
* **Persistent Data:** Upgrade from session-based memory to a persistent database to create a truly personalized companion.
* **Referral System:** Integrate with a database of mental health professionals to provide users with real-world support options.
