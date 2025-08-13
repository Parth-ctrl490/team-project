# app.py
# --------------------------------------------------------------------
# Indian Election Advisor 🗳 – Enhanced multilingual Flask + Groq streaming app
# Now provides comprehensive, detailed answers
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

# Model + temperature (slightly higher for more detailed responses)
GROQ_MODEL  = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.4))

# --------------------- ENHANCED SYSTEM PROMPT ------------------------
BASE_SYSTEM_PROMPT = """
You are 'भारतीय चुनाव सलाहकार' (Indian Election Advisor), an expert, comprehensive,
and highly knowledgeable AI assistant specializing exclusively in Indian election processes.

*Core Principles:*
1. **Comprehensive Coverage**: Provide detailed, thorough explanations that cover all aspects
   of the topic. Include step-by-step processes, requirements, timelines, and practical tips.

2. **Educational Depth**: Explain not just "what" but also "why" and "how". Include background
   context, legal basis, and practical implications of each process.

3. **Practical Guidance**: Always include:
   - Complete step-by-step procedures
   - Required documents and eligibility criteria
   - Timeline expectations and processing periods
   - Common challenges and how to overcome them
   - Alternative methods or backup options
   - Relevant contact information and resources

4. **Topic Focus**: Your expertise covers:
   - **Voter Registration**: Complete process, eligibility, documents, online/offline methods
   - **EPIC Cards**: Application, renewal, corrections, duplicates, status tracking
   - **Polling Stations**: Location finding, accessibility, facilities, booth-level information
   - **Election Schedules**: Dates, phases, notifications, important deadlines
   - **EVM/VVPAT**: Operation, security features, voter experience, troubleshooting
   - **Voter Rights & Duties**: Legal framework, complaint mechanisms, electoral laws
   - **Electoral Processes**: Candidate nomination, campaigning rules, counting procedures

5. **Safety Guard**: If asked about topics outside Indian elections (politics, candidates,
   opinions, other subjects), respond ONLY with the refusal text for the requested language.

6. **Welcome Protocol**: For "GREET_USER", provide a warm, detailed introduction with:
   - Your role and expertise areas
   - Comprehensive list of services you provide
   - How users can best utilize your knowledge
   - Encouraging tone with relevant emojis

7. **Response Structure**: 
   - Start with a brief overview
   - Provide detailed step-by-step information
   - Include practical tips and common scenarios
   - End with relevant resources or next steps
   - Use clear formatting with headers, bullets, and emphasis

8. **Formatting Standards**:
   - **Bold headers** for main sections
   - **Bold terms** for important concepts
   - Bullet points for processes and lists
   - Numbered steps for sequential procedures
   - Emojis for engagement (🗳️, 📝, 📍, 📅, ✅, ⚠️, 💡)
   - Clear paragraph breaks for readability

Remember: Your goal is to be the most comprehensive, helpful resource for Indian election
information. Users should leave with complete understanding and confidence to take action.

Now, follow these guidelines and respond in the language specified below.
---
"""

# --------------------- LANGUAGE CONFIGURATION ------------------------
LANGUAGE_CONFIG = {
    # --- Indo-Aryan ---------------------------------------------------
    'hi':  {"instruction": "Language: Respond in detailed, comprehensive Hindi. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *माफ करें, मैं केवल भारतीय चुनाव प्रक्रिया के बारे में जानकारी दे सकता हूं। कृपया मतदाता पंजीकरण, EPIC कार्ड, मतदान केंद्र, या अन्य चुनाव संबंधी प्रश्न पूछें।*"},
    
    'en':  {"instruction": "Language: Respond in detailed, comprehensive English. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *I can only provide information about Indian election processes. Please ask about voter registration, EPIC cards, polling stations, or other election-related topics.*"},
    
    'bn':  {"instruction": "Language: Respond in detailed, comprehensive Bengali. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *আমি শুধুমাত্র ভারতীয় নির্বাচন প্রক্রিয়া সম্পর্কে তথ্য দিতে পারি। দয়া করে ভোটার নিবন্ধন, EPIC কার্ড, বা নির্বাচন সংক্রান্ত প্রশ্ন করুন।*"},
    
    'mr':  {"instruction": "Language: Respond in detailed, comprehensive Marathi. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *मी फक्त भारतीय निवडणूक प्रक्रियेबद्दल माहिती देऊ शकतो. कृपया मतदार नोंदणी, EPIC कार्ड, किंवा निवडणूक संबंधित प्रश्न विचारा.*"},
    
    'gu':  {"instruction": "Language: Respond in detailed, comprehensive Gujarati. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *હું ફક્ત ભારતીય ચૂંટણી પ્રક્રિયા વિશે માહિતી આપી શકું છું. કૃપા કરીને મતદાર નોંધણી, EPIC કાર્ડ, અથવા ચૂંટણી સંબંધિત પ્રશ્નો પૂછો.*"},
    
    'pa':  {"instruction": "Language: Respond in detailed, comprehensive Punjabi. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *ਮੈਂ ਸਿਰਫ਼ ਭਾਰਤੀ ਚੋਣ ਪ੍ਰਕਿਰਿਆ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦੇ ਸਕਦਾ ਹਾਂ। ਕਿਰਪਾ ਕਰਕੇ ਵੋਟਰ ਰਜਿਸਟਰੇਸ਼ਨ, EPIC ਕਾਰਡ, ਜਾਂ ਚੋਣ ਸੰਬੰਧੀ ਸਵਾਲ ਪੁੱਛੋ।*"},
    
    'ur':  {"instruction": "Language: Respond in detailed, comprehensive Urdu (Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Main sirf Bharatiya election process ke bare mein maloomaat de sakta hun. Kripaya voter registration, EPIC card, ya election se mutalliq sawalat poochen.*"},
    
    'or':  {"instruction": "Language: Respond in detailed, comprehensive Odia. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *ମୁଁ କେବଳ ଭାରତୀୟ ନିର୍ବାଚନ ପ୍ରକ୍ରିୟା ବିଷୟରେ ସୂଚନା ଦେଇପାରେ। ଦୟାକରି ଭୋଟର ପଞ୍ଜୀକରଣ, EPIC କାର୍ଡ, କିମ୍ବା ନିର୍ବାଚନ ସଂପର୍କୀୟ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ।*"},
    
    'as':  {"instruction": "Language: Respond in detailed, comprehensive Assamese. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *মই কেৱল ভাৰতীয় নিৰ্বাচন প্ৰক্ৰিয়াৰ বিষয়ে তথ্য দিব পাৰো। দয়া কৰি ভোটাৰ পঞ্জীয়ন, EPIC কাৰ্ড, বা নিৰ্বাচন সম্পৰ্কীয় প্ৰশ্ন সোধক।*"},
    
    'ks':  {"instruction": "Language: Respond in detailed, comprehensive Kashmiri (Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Bi sirf Bharatiya election process baaras jaankaari dith shakaan. Meharbani karith voter registration, EPIC card ya election-related sawalaat puchiv.*"},
    
    'kok': {"instruction": "Language: Respond in detailed, comprehensive Konkani. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *हांव फकत भारतीय निवडणुकेच्या प्रक्रियेविशीं माहिती दिवंक शकतां. कृपया मतदार नोंदणी, EPIC कार्ड वा निवडणुकेशीं संबंदीत प्रस्न विचारात.*"},
    
    'sd':  {"instruction": "Language: Respond in detailed, comprehensive Sindhi (Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Mun sirf Bharatiya election process baare maloomaat ڏئي sakun ٿو. Meharbani kari voter registration, EPIC card ya election related sawal puchho.*"},
    
    'ne':  {"instruction": "Language: Respond in detailed, comprehensive Nepali. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *म केवल भारतीय चुनाव प्रक्रियाको बारेमा जानकारी दिन सक्छु। कृपया मतदाता दर्ता, EPIC कार्ड, वा चुनाव सम्बन्धी प्रश्नहरू सोध्नुहोस्।*"},
    
    'doi': {"instruction": "Language: Respond in detailed, comprehensive Dogri (Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Main sirf Bharatiya election process de baare jaanakari de sakda. Kripaya voter registration, EPIC card ya election related sawal pucho.*"},
    
    'brx': {"instruction": "Language: Respond in detailed, comprehensive Bodo. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Ang sirf Bharatiya election processni thangkha jaanakari kousak gathang. Kripa haba voter registration, EPIC card bage election related hola phusolai.*"},
    
    'sat': {"instruction": "Language: Respond in detailed, comprehensive Santali (Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Ing khali Bharatiya election process araete babodte paarkom. Daaya katet voter registration, EPIC card kimba election okoyre kushiyako kuli me.*"},

    # --- Dravidian ----------------------------------------------------
    'ta':  {"instruction": "Language: Respond in detailed, comprehensive Tamil. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *நான் இந்திய தேர்தல் செயல்முறைகளைப் பற்றி மட்டுமே தகவல் அளிக்க முடியும். தயவுசெய்து வாக்காளர் பதிவு, EPIC அட்டை, அல்லது தேர்தல் தொடர்பான கேள்விகளைக் கேளுங்கள்.*"},
    
    'te':  {"instruction": "Language: Respond in detailed, comprehensive Telugu. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *నేను భారతీయ ఎన్నికల ప్రక్రియల గురించి మాత్రమే సమాచారం అందించగలను. దయచేసి ఓటరు నమోదు, EPIC కార్డు, లేదా ఎన్నికల సంబంధిత ప్రశ్నలు అడగండి.*"},
    
    'kn':  {"instruction": "Language: Respond in detailed, comprehensive Kannada. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *ನಾನು ಭಾರತೀಯ ಚುನಾವಣಾ ಪ್ರಕ್ರಿಯೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲೆ. ದಯವಿಟ್ಟು ಮತದಾರ ನೋಂದಣಿ, EPIC ಕಾರ್ಡ್, ಅಥವಾ ಚುನಾವಣೆ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.*"},
    
    'ml':  {"instruction": "Language: Respond in detailed, comprehensive Malayalam. Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *എനിക്ക് ഇന്ത്യൻ തിരഞ്ഞെടുപ്പ് പ്രക്രിയകളെ കുറിച്ച് മാത്രമേ വിവരങ്ങൾ നൽകാൻ കഴിയൂ. ദയവായി വോട്ടർ രജിസ്ട്രേഷൻ, EPIC കാർഡ്, അല്ലെങ്കിൽ തിരഞ്ഞെടുപ്പ് സംബന്ധിച്ച ചോദ്യങ്ങൾ ചോദിക്കുക.*"},

    # --- Tibeto-Burman -----------------------------------------------
    'mni': {"instruction": "Language: Respond in detailed, comprehensive Manipuri (Meitei, Roman script). Provide thorough explanations with complete procedures and practical guidance.",
            "refusal": "🚫 *Ei sirf Bharatiya election processgi maramda information piba ngammi. Charaakkhro voter registration, EPIC card nattraga election mareldaba wahang hangbi puraaku.*"},
}

# Helper to build the full system prompt for a language
def get_system_prompt(lang_code: str) -> str:
    config = LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['hi'])
    return f"{BASE_SYSTEM_PROMPT}\n{config['instruction']}"

# --------------------------- FEEDBACK ROUTE --------------------------
@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback submission"""
    try:
        data = request.json or {}
        user_message = data.get('user_message', '')
        bot_response = data.get('bot_response', '')
        feedback_type = data.get('feedback_type', '')  # 'positive' or 'negative'
        comment = data.get('comment', '')
        language = data.get('language', 'hi')
        
        # Log feedback (in production, save to database)
        app.logger.info(f"Feedback received: {feedback_type} - Lang: {language}")
        app.logger.info(f"User query: {user_message[:100]}...")
        app.logger.info(f"Comment: {comment}")
        
        # Here you could save to database, send to analytics, etc.
        # For now, just acknowledge receipt
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

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
                max_tokens  = 2048,  # Increased for more detailed responses
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
            # Limit chat history to prevent context overflow
            if len(chat_history) > 10:  # Keep last 10 exchanges
                chat_history = chat_history[-10:]
            session['chat_history'] = chat_history

        return Response(
            stream_with_context(generate_chunks()),
            content_type='text/plain'
        )

    except Exception as e:
        app.logger.error(f"Error calling Groq API: {e}")
        return jsonify({'error': 'An error occurred while communicating with the AI.'}), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    session.pop('chat_history', None)
    return jsonify({'status': 'success', 'message': 'Chat history cleared.'})

# --------------------------- RUN APP ---------------------------------
if __name__ == '__main__':
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.run(debug=not is_production, port=5001)
