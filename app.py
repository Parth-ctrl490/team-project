# app.py
# --------------------------------------------------------------------
# Indian Election Advisor 🗳️ – multilingual Flask + Groq streaming app
# --------------------------------------------------------------------
import os
from flask import (
    Flask, render_template, request,
    jsonify, session, Response, stream_with_context
)
from groq import Groq
from dotenv import load_dotenv
from langdetect import detect

# ----------------------- INITIALISATION ------------------------------
load_dotenv()                                           # Load .env vars
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY', 'a-very-secret-key-for-development'
)

# Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ Missing GROQ_API_KEY in .env file")
client = Groq(api_key=api_key)

# Model + temperature
GROQ_MODEL  = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))

# --------------------- ADVANCED SYSTEM PROMPT ------------------------
# This prompt forces the structured reasoning. It remains unchanged.
BASE_SYSTEM_PROMPT = """
You are 'भारतीय चुनाव सलाहकार' (Indian Election Advisor), an expert AI. Your primary directive is to provide accurate information on Indian elections and aggressively counter misinformation.

For EVERY user query, you MUST perform a rigorous two-step process internally before generating a response.

**STEP 1: Internal Chain-of-Thought Analysis (DO NOT show this to the user)**
You will first reason about the user's query by thinking through these points:
1.  **PREMISE:** What is the core assumption or claim in the user's query?
2.  **VERIFICATION:** Is this premise factually correct according to official Indian election rules? (e.g., "Fact-check: The premise 'voting age is 16' is FALSE. The official age is 18.")
3.  **CLASSIFICATION:** Based on your verification, classify the query into ONE of the following categories:
    *   `MISINFORMATION`: If the user's premise is factually incorrect or dangerously misleading about the election process.
    *   `VALID_QUESTION`: If the user is asking a legitimate, on-topic question without any false premise.
    *   `OFF_TOPIC`: If the query is safe but not about the Indian election process (e.g., politics, opinions, sports).
    *   `GREETING`: Only for the special "GREET_USER" prompt.

**STEP 2: Generate Structured Response**
After your internal analysis, you MUST provide your final output in the following strict format, and nothing else:
` [Your Classification from Step 1]`
` [Your user-facing response, based on the rules below]`

**Response Rules based on Classification:**
*   If `CLASSIFICATION` is `MISINFORMATION`, the `RESPONSE` MUST use the following corrective template: {MISINFO_RESPONSE_TEMPLATE}
*   If `CLASSIFICATION` is `VALID_QUESTION`, the `RESPONSE` MUST be a helpful, concise answer to the user's query.
*   If `CLASSIFICATION` is `OFF_TOPIC`, the `RESPONSE` MUST be ONLY the specific refusal text.
*   If `CLASSIFICATION` is `GREETING`, the `RESPONSE` MUST be a warm, welcoming introduction.

Use markdown and emojis in your `RESPONSE` as appropriate. You will now strictly follow this two-step process for all queries, in the language specified below.
---
"""
# --------------------- Misinformation Response Template ----------------
MISINFO_RESPONSE_TEMPLATE = (
    "🚫 **माफ करें, यह जानकारी गलत है।**\n\n"
    "यहाँ सही जानकारी है: {correct_info}\n\n"
    "आपके प्रश्न का सही उत्तर: {correct_answer}"
)


# --------------------- LANGUAGE CONFIGURATION ------------------------
# 22 constitutionally recognised Indian languages + English.
# ISO-639-1 codes used when available; otherwise common 3-letter codes.
LANGUAGE_CONFIG = {
    # --- Indo-Aryan ---------------------------------------------------
    'hi':  {"instruction": "Language: Respond in simple, clear Hindi.",
            "refusal": "🚫 **माफ करें, मैं इस बारे में नहीं जानता।**"},
    'en':  {"instruction": "Language: Respond in simple, clear English.",
            "refusal": "🚫 **Sorry, I do not have information on that topic.**"},
    'bn':  {"instruction": "Language: Respond in simple, clear Bengali.",
            "refusal": "🚫 **দুঃখিত, আমি এই বিষয়ে জানি না।**"},
    'mr':  {"instruction": "Language: Respond in simple, clear Marathi.",
            "refusal": "🚫 **माफ करा, मला याबद्दल माहिती नाही.**"},
    'gu':  {"instruction": "Language: Respond in simple, clear Gujarati.",
            "refusal": "🚫 **માફ કરશો, મને આ વિષય વિશે માહિતી નથી.**"},
    'pa':  {"instruction": "Language: Respond in simple, clear Punjabi.",
            "refusal": "🚫 **ਮਾਫ ਕਰੋ, ਮੈਨੂੰ ਇਸ ਬਾਰੇ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।**"},
    'ur':  {"instruction": "Language: Respond in simple, clear Urdu (Roman script).",
            "refusal": "🚫 **Maaf kijiye, mujhe is bare mein maloomaat nahin hai.**"},
    'or':  {"instruction": "Language: Respond in simple, clear Odia.",
            "refusal": "🚫 **ମାଫ କରନ୍ତୁ, ମୋତେ ଏହା ବିଷୟରେ ଜଣା ନାହିଁ।**"},
    'as':  {"instruction": "Language: Respond in simple, clear Assamese.",
            "refusal": "🚫 **দুঃখিত, মই এই বিষয়ে জানো নে।**"},
    'ks':  {"instruction": "Language: Respond in simple, clear Kashmiri (Roman script).",
            "refusal": "🚫 **Maaf kariv, me chu yim baareh chu na jaanan.**"},
    'kok': {"instruction": "Language: Respond in simple, clear Konkani.",
            "refusal": "🚫 **माका माफी करा, ह्या विषयाची माहिती नाका.**"},
    'sd':  {"instruction": "Language: Respond in simple, clear Sindhi (Roman script).",
            "refusal": "🚫 **Maaf kajo, munhnje koluyang aahe koi maloomaat na-ahe.**"},
    'ne':  {"instruction": "Language: Respond in simple, clear Nepali.",
            "refusal": "🚫 **माफ गर्नुस्, यो विषयमा जानकारी छैन।**"},
    'doi': {"instruction": "Language: Respond in simple, clear Dogri (Roman script).",
            "refusal": "🚫 **Maaf karo, maini yo baare koi jaanakari ni ae.**"},
    'brx': {"instruction": "Language: Respond in simple, clear Bodo.",
            "refusal": "🚫 **Khara matha, angni dangoria mwjang onsilai.**"},
    'sat': {"instruction": "Language: Respond in simple, clear Santali (Roman script).",
            "refusal": "🚫 **Maaph kana, chenget menak’ ‘romreya hor nai.**"},

    # --- Dravidian ----------------------------------------------------
    'ta':  {"instruction": "Language: Respond in simple, clear Tamil.",
            "refusal": "🚫 **மன்னிக்கவும், எனக்கு அதைப் பற்றித் தெரியாது.**"},
    'te':  {"instruction": "Language: Respond in simple, clear Telugu.",
            "refusal": "🚫 **క్షమించండి, నాకు ఆ విషయం తెలియదు.**"},
    'kn':  {"instruction": "Language: Respond in simple, clear Kannada.",
            "refusal": "🚫 **ಕ್ಷಮಿಸಿ, ನನಗೆ ಈ ವಿಷಯದ ಬಗ್ಗೆ ಮಾಹಿತಿ ಇಲ್ಲ.**"},
    'ml':  {"instruction": "Language: Respond in simple, clear Malayalam.",
            "refusal": "🚫 **ക്ഷമിക്കുക, എനിക്ക് ഈ വിഷയത്തെക്കുറിച്ച് അറിവില്ല.**"},

    # --- Tibeto-Burman -----------------------------------------------
    'mni': {"instruction": "Language: Respond in simple, clear Manipuri (Meitei, Roman script).",
            "refusal": "🚫 **Chaorakkhou, eigi makhada chennabadi khangdrabadi nattaba.**"},
}

# Helper to build the full system prompt for a language
def get_system_prompt(lang_code: str) -> str:
    config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['hi'])
    return f"{BASE_SYSTEM_PROMPT}\n{config['instruction']}"

# --------------------------- ROUTES ----------------------------------
@app.route('/')
def home():
    session.pop('chat_history', None)      # Reset history on fresh load
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()

    # Validate message
    if not user_message:
        return jsonify({'error': 'Invalid message.'}), 400

    # Language choice: frontend > automatic detection
    frontend_lang = data.get("lang")
    detected_lang  = detect(user_message) if not frontend_lang else None
    lang_code = (frontend_lang or detected_lang or 'hi').lower()
    if lang_code not in LANGUAGE_CONFIG:
        lang_code = 'hi'                  # safe fallback

    system_prompt = get_system_prompt(lang_code)
    chat_history  = session.get('chat_history', [])

    # First-message special handling
    if not chat_history and user_message == "GREET_USER":
        messages_for_api = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "GREET_USER"}
        ]
    else:
        messages_for_api = [
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "user", "content": user_message}
        ]

    # ------------------ Streaming response from Groq -----------------
    try:
        def generate_chunks():
            full_resp = ""
            stream = client.chat.completions.create(
                messages    = messages_for_api,
                model       = GROQ_MODEL,
                temperature = TEMPERATURE,
                stream      = True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    full_resp += token
                    yield token

            # Persist chat after full answer is produced
            chat_history.extend([
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": full_resp}
            ])
            session['chat_history'] = chat_history

        return Response(
            stream_with_context(generate_chunks()),
            content_type='text/plain'
        )

    except Exception as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({'error': 'An error occurred while communicating with the AI.'}), 500
    
# --- MODIFIED FEEDBACK ROUTE ---
@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    Receives detailed feedback from the frontend, including an optional comment,
    and logs it to the console.
    """
    try:
        data = request.json or {}
        user_message = data.get('user_message')
        bot_response = data.get('bot_response')
        feedback_type = data.get('feedback_type') # 'positive' or 'negative'
        comment = data.get('comment', 'No comment provided.') # NEW: Get the comment
        lang = data.get('language')

        print("\n--- ✅ DETAILED FEEDBACK RECEIVED ---")
        print(f"Language: {lang}")
        print(f"User's Question: {user_message}")
        print(f"AI's Response: {bot_response}")
        print(f"Feedback Given: {feedback_type.upper()}")
        print(f"User Comment: {comment}") # NEW: Log the comment
        print("------------------------------------\n")
        
        return jsonify({'status': 'success', 'message': 'Feedback has been recorded. Thank you!'}), 200

    except Exception as e:
        app.logger.error(f"Error processing feedback: {e}")
        return jsonify({'status': 'error', 'message': 'Could not process feedback.'}), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    session.pop('chat_history', None)
    return jsonify({'status': 'success', 'message': 'Chat history cleared.'})

# --------------------------- RUN APP ---------------------------------
if __name__ == '__main__':
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.run(debug=not is_production, port=5001)
