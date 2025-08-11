# app.py

import os
from flask import (
    Flask, render_template, request, jsonify,
    session, Response, stream_with_context
)
from groq import Groq
from dotenv import load_dotenv
from langdetect import detect

# --- SETUP ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# Secret key for session management (important for production)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY',
    'a-very-secret-key-for-development'
)

# Groq API Client setup
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ Missing GROQ_API_KEY in .env file")
client = Groq(api_key=api_key)

# Model and temperature configuration
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))

# --- ADVANCED SYSTEM PROMPT ---
BASE_SYSTEM_PROMPT = """
You are 'भारतीय चुनाव सलाहकार' (Indian Election Advisor), an expert, helpful, and concise AI assistant.
Your knowledge is strictly limited to the Indian election processes.

*Core Rules:*
1. *Topic Focus:* Only answer questions about Indian voter registration, EPIC cards, EVM/VVPAT usage, finding polling booths, election schedules, and voter rights/duties.
2. *Safety Guard:* If asked about anything else (e.g., politics, specific candidates, opinions, sports, movies), you MUST respond with ONLY the specific refusal text for the requested language and nothing more.
3. *Welcome Message:* If the user's first message is "GREET_USER", you MUST introduce yourself warmly, state your purpose, and list your main capabilities in a friendly, bulleted list. Use emojis to make it engaging.
4. *Formatting:* Keep answers informative but concise. Use markdown for clarity:
    - Use **bold text** for headings and important terms.
    - Use bullet points (• or numbered lists) for steps and lists.
    - Use emojis (e.g., 🗳, 📝, 📍, 📅).

Now, follow these rules and respond in the language specified below.
---
"""

# Language-specific instructions and refusal messages
LANGUAGE_CONFIG = {
    'hi': {"instruction": "Language: Respond in simple, clear Hindi.", "refusal": "🚫 *माफ करें, मैं इस बारे में नहीं जानता।*"},
    'en': {"instruction": "Language: Respond in simple, clear English.", "refusal": "🚫 *Sorry, I do not have information on that topic.*"},
    'bn': {"instruction": "Language: Respond in simple, clear Bengali.", "refusal": "🚫 *দুঃখিত, আমি এই বিষয়ে জানি না।*"},
    'ta': {"instruction": "Language: Respond in simple, clear Tamil.", "refusal": "🚫 *மன்னிக்கவும், எனக்கு அதைப் பற்றித் தெரியாது.*"},
    'mr': {"instruction": "Language: Respond in simple, clear Marathi.", "refusal": "🚫 *माफ करा, मला याबद्दल माहिती नाही.*"}
}

def get_system_prompt(lang_code):
    """Builds the full system prompt by combining base rules with language-specific instructions."""
    config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['hi'])
    return f"{BASE_SYSTEM_PROMPT}\n{config['instruction']}"

# --- ROUTES ---

@app.route('/')
def home():
    session.pop('chat_history', None)
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")

    # Determine language
    lang_from_frontend = data.get("lang")
    lang_code = lang_from_frontend or (detect(user_message) if user_message else 'hi')
    lang_code = lang_code if lang_code in LANGUAGE_CONFIG else 'hi'

    if not user_message:
        return jsonify({'error': 'Invalid message.'}), 400

    system_prompt = get_system_prompt(lang_code)
    chat_history = session.get('chat_history', [])

    # Special handling for first-time greeting
    if not chat_history and user_message == "GREET_USER":
        messages_for_api = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "GREET_USER"}
        ]
    else:
        messages_for_api = [
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "user", "content": user_message}
        ]

    try:
        def generate_chunks():
            """Stream the AI's response back to the client."""
            full_response = ""
            stream = client.chat.completions.create(
                messages=messages_for_api,
                model=GROQ_MODEL,
                temperature=TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # Save updated chat history
            chat_history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_response}
            ])
            session['chat_history'] = chat_history

        return Response(stream_with_context(generate_chunks()), content_type='text/plain')

    except Exception as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({'error': 'An error occurred while communicating with the AI.'}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """Clears the chat history from the session."""
    session.pop('chat_history', None)
    return jsonify({'status': 'success', 'message': 'Chat history cleared.'})

# --- RUN APP ---
if __name__ == '__main__':
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.run(debug=not is_production, port=5001)
