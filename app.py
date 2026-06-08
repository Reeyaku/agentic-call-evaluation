import streamlit as st
import pandas as pd
import re
import tempfile
import json
import os
import time
from textblob import TextBlob
from google import genai

try:
    import whisper
except Exception:
    whisper = None


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Agentic AI Call Evaluation System",
    page_icon="📞",
    layout="wide"
)


# ---------------- CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%);
}

.hero {
    background: linear-gradient(135deg, #111827, #243447, #374151);
    padding: 36px;
    border-radius: 26px;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(17,24,39,0.22);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #d1d5db;
    font-size: 18px;
    line-height: 1.6;
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #1f2937;
    margin-top: 8px;
    margin-bottom: 10px;
}

.badge-success {
    background-color: #dcfce7;
    color: #166534;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.badge-warning {
    background-color: #fef3c7;
    color: #92400e;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.badge-danger {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-weight: 700;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CONSTANTS ----------------

CONFIRMATION_WORDS = [
    "yes", "correct", "right", "confirmed", "true", "exactly",
    "haan", "ha", "ji", "sahi", "bilkul", "theek hai", "okay"
]

NEGATIVE_WORDS = [
    "busy", "later", "call later", "not interested", "wrong number",
    "stop calling", "already told", "why again", "frustrated",
    "irritated", "angry", "nahi", "baad mein", "mat karo",
    "not a business", "don't know", "samajh nahi"
]

BOT_QUESTION_WORDS = [
    "confirm", "verify", "may i know", "can you", "could you",
    "please", "business", "company", "address", "mobile",
    "number", "timing", "working hours"
]


# ---------------- WHISPER TRANSCRIPTION ----------------

@st.cache_resource
def load_model(model_size):
    return whisper.load_model(model_size)


def transcribe_audio(uploaded_file, transcription_mode):
    if whisper is None:
        st.error("Whisper is not installed. Please check requirements.txt.")
        return ""

    uploaded_file.seek(0)
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    if transcription_mode == "English":
        model_size = "base"
        language = None
        task = "translate"
        prompt = (
            "Translate this Indian business verification call into clear English. "
            "Preserve company names, mobile numbers, addresses, and working hours accurately. "
            "Do not assume missing business information."
        )

    elif transcription_mode == "Hindi":
        model_size = "base"
        language = "hi"
        task = "transcribe"
        prompt = (
            "इस कॉल को हिंदी देवनागरी लिपि में लिखें। "
            "कंपनी का नाम, मोबाइल नंबर, पता और काम करने का समय सही रखें।"
        )

    else:
        model_size = "base"
        language = "hi"
        task = "transcribe"
        prompt = (
            "Transcribe Hindi speech in Roman Hindi / Hinglish. "
            "Example: main neha justdial se baat kar rahi hu. "
            "Preserve company names, mobile numbers, addresses and timings."
        )

    model = load_model(model_size)

    result = model.transcribe(
        temp_audio_path,
        language=language,
        task=task,
        fp16=False,
        initial_prompt=prompt
    )

    return result.get("text", "").strip()


# ---------------- GEMINI EVALUATION ----------------

def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error("Gemini API key is missing. Add GEMINI_API_KEY inside .streamlit/secrets.toml.")
        st.stop()


def clean_json_response(content):
    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    return json.loads(content)


def gemini_evaluate_call(transcript, transcription_mode):
    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)

    if transcription_mode == "Hindi":
        field_language_rules = """
For Hindi mode:
- Field names must be in Hindi:
  कंपनी का नाम, मोबाइल नंबर, पता, काम करने का समय
- Status values must be in Hindi:
  पुष्टि हुई, आंशिक पुष्टि, पुष्टि नहीं हुई
- Extracted values, evidence, and summary must be in Hindi.
"""
        field_1 = "कंपनी का नाम"
        field_2 = "मोबाइल नंबर"
        field_3 = "पता"
        field_4 = "काम करने का समय"
        status_values = "पुष्टि हुई / आंशिक पुष्टि / पुष्टि नहीं हुई"
        summary_language = "Hindi"
    else:
        field_language_rules = """
For English/Hinglish mode:
- Field names must be in English:
  Company Name, Mobile Number, Address, Working Hours
- Status values must be in English:
  Confirmed, Partially Confirmed, Not Confirmed
- Extracted values, evidence, and summary must be in English.
"""
        field_1 = "Company Name"
        field_2 = "Mobile Number"
        field_3 = "Address"
        field_4 = "Working Hours"
        status_values = "Confirmed / Partially Confirmed / Not Confirmed"
        summary_language = "English"

    prompt = f"""
You are a strict QA analyst for an Agentic AI business verification call.

Analyze the transcript and return ONLY valid JSON.

{field_language_rules}

CRITICAL CONVERSATION LOGIC:
- Treat the call as a sequence of bot questions and vendor/user responses.
- Do NOT mark a field as Confirmed just because the bot mentioned the value.
- Evidence must come from the vendor/user response, not only from the bot question.
- If the bot asks a verification question and the vendor/user does not clearly answer that question, mark that field as Not Confirmed.
- If the bot asks "Is your business name Hotel Shauri Wada correct?" and the vendor says something unrelated like "Why are you calling?" or "I already gave this information", then Company Name must be Not Confirmed.
- If the vendor/user says "yes", "correct", "haan", "sahi hai", "bilkul", or equivalent immediately after the bot's field question, then mark the field as Confirmed.
- If the vendor/user gives a different corrected value, use the corrected value and mark the field as Confirmed.
- If the vendor/user gives incomplete or unclear information, mark the field as Partially Confirmed or Not Confirmed depending on clarity.

STRICT RULES:
1. Do NOT assume values.
2. A field is Confirmed ONLY if the user/vendor clearly confirms, corrects, or directly provides it.
3. A value mentioned only by the bot must never be marked Confirmed.
4. If unclear, missing, guessed, or only said by the bot, mark it as Not Confirmed.
5. A field can be confirmed only when vendor directly provides the value OR vendor explicitly confirms the value after the bot asks.
6. If the bot says a value but the user/vendor does not clearly confirm it, mark it as Not Confirmed.
7.If vendor does not give response, expresses confusion or gives unrelated response, mark it as Not Confirmed.
8. Use "-" for missing values.
9. Add short evidence for every field.
10. Evaluate ONLY these 4 fields:
   - Company Name
   - Mobile Number
   - Address
   - Working Hours
11. Do NOT include Contact Person.
12. Do NOT include Business Category.

Transcript:
{transcript}

Return JSON exactly in this structure:

{{
  "fields_confirmed": [
    {{"Field": "{field_1}", "Status": "{status_values}", "Extracted Value": "value or -", "Evidence": "Reason"}},
    {{"Field": "{field_2}", "Status": "{status_values}", "Extracted Value": "value or -", "Evidence": "Reason"}},
    {{"Field": "{field_3}", "Status": "{status_values}", "Extracted Value": "value or -", "Evidence": "Reason"}},
    {{"Field": "{field_4}", "Status": "{status_values}", "Extracted Value": "value or -", "Evidence": "Reason"}}
  ],
  "bot_quality": {{
    "rating": "Good / Average / Poor",
    "reason": "Reason in English",
    "parameters": [
      ["Followed Conversation Flow", "Yes / No / Partial"],
      ["Asked Relevant Verification Questions", "Yes / No / Partial"],
      ["Handled Objections Properly", "Yes / No / Partial / Not Applicable"],
      ["Avoided Repetition / Confusion", "Yes / No / Partial"],
      ["Successfully Completed Data Collection", "x/4 fields confirmed"],
      ["Maintained Professional Tone", "Yes / No / Partial"]
    ]
  }},
  "user_quality": {{
    "rating": "Good / Average / Poor",
    "sentiment": "Positive / Neutral / Negative / Frustrated / Confused",
    "reason": "Reason in English",
    "parameters": [
      ["Cooperated During Conversation", "Yes / No / Partial"],
      ["Stayed Relevant to Discussion", "Yes / No / Partial"],
      ["Provided Clear Responses", "Yes / No / Partial"],
      ["Shared Required Business Details", "x/4 fields confirmed"],
      ["Frustration / Confusion Detected", "Yes / No"]
    ]
  }},
  "overall_call_status": "Successful / Partially Successful / Failed / Follow-up Required",
  "summary": "Brief professional summary in {summary_language}"
}}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            break

        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                if attempt < 2:
                    st.warning("Gemini quota limit reached. Waiting 20 seconds and retrying...")
                    time.sleep(20)
                else:
                    st.error("Gemini quota exceeded. Please wait for quota reset or enable billing.")
                    st.stop()
            elif "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < 2:
                    st.warning("Gemini is temporarily busy. Waiting 10 seconds and retrying...")
                    time.sleep(10)
                else:
                    st.error("Gemini is currently unavailable. Please try again after a few minutes.")
                    st.stop()
            else:
                raise e

    return clean_json_response(response.text)


# ---------------- TEXT PROCESSING ----------------

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_question_like(sentence):
    lower = sentence.lower().strip()
    question_markers = [
        "?", "kya", "can you", "could you", "please confirm",
        "confirm", "verify", "sahi hai", "right hai",
        "is this correct", "is it correct", "hai kya"
    ]
    return any(marker in lower for marker in question_markers)


# ---------------- LOCAL FIELD EXTRACTION FOR HINGLISH MODE ----------------

def extract_phone(text):
    pattern = r'(\+91[\-\s]?)?[6-9]\d{9}'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_company_name(text):
    patterns = [
        r"speaking with ([A-Za-z0-9\s&]+)",
        r"company name is ([A-Za-z0-9\s&]+)",
        r"business name is ([A-Za-z0-9\s&]+)",
        r"business ka naam ([A-Za-z0-9\s&]+)",
        r"company ka naam ([A-Za-z0-9\s&]+)"
    ]

    for sentence in split_into_sentences(text):
        if is_question_like(sentence):
            continue

        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = re.split(r"[.,!?]", value)[0].strip()
                if value and len(value.split()) <= 6:
                    return value

    return None


def extract_address(text):
    keywords = [
        "address", "located", "near", "landmark",
        "road", "street", "nagar", "layout", "area",
        "bangalore", "bengaluru", "mumbai", "pune", "delhi",
        "hyderabad", "chennai", "kolkata", "floor", "block",
        "sector", "phase", "main road", "cross road"
    ]

    reject_phrases = [
        "samajh nahi",
        "samajh mein",
        "kya bol",
        "do not understand",
        "what are you saying",
        "phone",
        "number"
    ]

    for sentence in split_into_sentences(text):
        lower_sentence = sentence.lower()

        if is_question_like(sentence):
            continue

        if any(reject in lower_sentence for reject in reject_phrases):
            continue

        if any(k in lower_sentence for k in keywords) and len(sentence.split()) >= 4:
            return sentence.strip()

    return None


def extract_timings(text):
    patterns = [
        r'\d{1,2}\s?(AM|PM|am|pm)\s?(to|-)\s?\d{1,2}\s?(AM|PM|am|pm)',
        r'\d{1,2}\s?(baje|bajey)\s?(se|to|-)\s?\d{1,2}\s?(baje|bajey)'
    ]

    for sentence in split_into_sentences(text):
        if is_question_like(sentence):
            continue

        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                return match.group(0)

    return None


def field_status(value):
    if not value:
        return "Not Confirmed"
    return "Partially Confirmed"


def extract_fields(transcript):
    company = extract_company_name(transcript)
    phone = extract_phone(transcript)
    address = extract_address(transcript)
    timings = extract_timings(transcript)

    return [
        {
            "Field": "Company Name",
            "Status": field_status(company),
            "Extracted Value": company or "-",
            "Evidence": "Detected from local Hinglish transcript." if company else "No clear company name found."
        },
        {
            "Field": "Mobile Number",
            "Status": field_status(phone),
            "Extracted Value": phone or "-",
            "Evidence": "Detected valid Indian mobile number." if phone else "No valid mobile number found."
        },
        {
            "Field": "Address",
            "Status": field_status(address),
            "Extracted Value": address or "-",
            "Evidence": "Detected address-like phrase." if address else "No clear address found."
        },
        {
            "Field": "Working Hours",
            "Status": field_status(timings),
            "Extracted Value": timings or "-",
            "Evidence": "Detected working-hour pattern." if timings else "No clear working hours found."
        }
    ]


# ---------------- LOCAL QUALITY EVALUATION FOR HINGLISH MODE ----------------

def sentiment_label(text):
    polarity = TextBlob(text).sentiment.polarity

    if any(word in text.lower() for word in NEGATIVE_WORDS):
        return "Negative / Frustrated"

    if polarity > 0.15:
        return "Positive"
    elif polarity < -0.15:
        return "Negative"
    else:
        return "Neutral"


def evaluate_bot(transcript, fields):
    confirmed = sum(1 for f in fields if f["Status"] == "Confirmed")
    partial = sum(1 for f in fields if f["Status"] == "Partially Confirmed")

    relevant_questions = any(word in transcript.lower() for word in BOT_QUESTION_WORDS)
    repetition_count = transcript.lower().count("confirm") + transcript.lower().count("again")
    professional_tone = any(word in transcript.lower() for word in ["please", "thank you", "thanks", "kindly"])

    score = 0

    if relevant_questions:
        score += 25
    if professional_tone:
        score += 20
    if repetition_count <= 4:
        score += 20
    if confirmed + partial >= 3:
        score += 25
    elif confirmed + partial >= 2:
        score += 15
    else:
        score += 5

    if score >= 75:
        rating = "Good"
    elif score >= 45:
        rating = "Average"
    else:
        rating = "Poor"

    checks = [
        ["Followed Conversation Flow", "Yes" if relevant_questions else "Needs Improvement"],
        ["Asked Relevant Verification Questions", "Yes" if relevant_questions else "No"],
        ["Handled Objections Properly", "Partial" if any(w in transcript.lower() for w in NEGATIVE_WORDS) else "Not Applicable"],
        ["Avoided Repetition / Confusion", "Yes" if repetition_count <= 4 else "No"],
        ["Successfully Completed Data Collection", f"{confirmed}/4 fields confirmed"],
        ["Maintained Professional Tone", "Yes" if professional_tone else "Needs Improvement"]
    ]

    reason = (
        f"The bot captured {confirmed}/4 confirmed fields and {partial}/4 partially confirmed fields. "
        f"It was rated {rating.lower()} based on flow, relevant questions, repetition control, "
        f"professional tone, and business data collection."
    )

    return rating, score, reason, checks


def evaluate_user(transcript, fields):
    sentiment = sentiment_label(transcript)
    confirmed = sum(1 for f in fields if f["Status"] == "Confirmed")
    partial = sum(1 for f in fields if f["Status"] == "Partially Confirmed")
    frustration = any(word in transcript.lower() for word in NEGATIVE_WORDS)

    score = 0

    if confirmed + partial >= 3:
        score += 40
    elif confirmed + partial >= 2:
        score += 25
    else:
        score += 10

    if not frustration:
        score += 30
    else:
        score += 10

    if len(transcript.split()) > 25:
        score += 20
    else:
        score += 10

    if sentiment in ["Positive", "Neutral"]:
        score += 10
    else:
        score += 5

    if score >= 75:
        rating = "Good"
    elif score >= 45:
        rating = "Average"
    else:
        rating = "Poor"

    checks = [
        ["Cooperated During Conversation", "Yes" if confirmed + partial >= 2 else "Needs Improvement"],
        ["Stayed Relevant to Discussion", "Yes" if len(transcript.split()) > 20 else "Partial"],
        ["Provided Clear Responses", "Yes" if confirmed + partial >= 2 else "Needs Improvement"],
        ["Shared Required Business Details", f"{confirmed}/4 fields confirmed"],
        ["Frustration / Confusion Detected", "Yes" if frustration else "No"]
    ]

    reason = (
        f"The user was rated {rating.lower()}. Sentiment was detected as {sentiment}. "
        f"The user contributed to capturing {confirmed}/4 confirmed fields and {partial}/4 partially confirmed fields."
    )

    if frustration:
        reason += " Some resistance, confusion, or frustration was detected during the call."

    return rating, score, reason, sentiment, checks


def call_status(fields):
    confirmed = sum(1 for f in fields if f["Status"] in ["Confirmed", "पुष्टि हुई"])
    partial = sum(1 for f in fields if f["Status"] in ["Partially Confirmed", "आंशिक पुष्टि"])

    if confirmed >= 4:
        return "Successful"
    elif confirmed + partial >= 2:
        return "Partially Successful"
    else:
        return "Failed / Follow-up Required"


def create_summary(status, bot_rating, user_rating, fields):
    confirmed = [f["Field"] for f in fields if f["Status"] in ["Confirmed", "पुष्टि हुई"]]
    partial = [f["Field"] for f in fields if f["Status"] in ["Partially Confirmed", "आंशिक पुष्टि"]]
    missing = [f["Field"] for f in fields if f["Status"] in ["Not Confirmed", "पुष्टि नहीं हुई"]]

    return f"""
The call was classified as **{status}**.

The bot was rated **{bot_rating}** based on conversation flow, relevant verification questions, objection handling, repetition control, data collection completion, and professional tone.

The user was rated **{user_rating}** based on cooperation, response clarity, relevance, business detail sharing, and signs of frustration or confusion.

**Confirmed Fields:** {", ".join(confirmed) if confirmed else "None"}  
**Partially Confirmed Fields:** {", ".join(partial) if partial else "None"}  
**Missing Fields:** {", ".join(missing) if missing else "None"}
"""


# ---------------- UTILITY FUNCTIONS ----------------

def calculate_completion(fields):
    confirmed_count = sum(1 for f in fields if f["Status"] in ["Confirmed", "पुष्टि हुई"])
    partial_count = sum(1 for f in fields if f["Status"] in ["Partially Confirmed", "आंशिक पुष्टि"])
    total_fields = len(fields)

    if total_fields == 0:
        return 0, 0

    completion = int(((confirmed_count + 0.5 * partial_count) / total_fields) * 100)
    return completion, confirmed_count


def status_style(val):
    if val in ["Confirmed", "पुष्टि हुई"]:
        return "background-color: #dcfce7; color: #166534; font-weight: 700;"
    elif val in ["Partially Confirmed", "आंशिक पुष्टि"]:
        return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"


def display_status_badge(status):
    if status == "Successful":
        st.markdown(f'<span class="badge-success">{status}</span>', unsafe_allow_html=True)
    elif status == "Partially Successful":
        st.markdown(f'<span class="badge-warning">{status}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-danger">{status}</span>', unsafe_allow_html=True)


# ---------------- UI ----------------

st.markdown("""
<div class="hero">
    <h1>📞 Agentic AI Call Evaluation System</h1>
    <p>
    Upload a business verification call recording to generate transcription, verified business fields,
    bot quality, user quality, sentiment analysis, call summary, and downloadable QA reports.
    </p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-title">🎧 Upload Call Recording</div>', unsafe_allow_html=True)

    transcription_mode = st.radio(
        "Select Transcription Output Format",
        ["English", "Hindi", "Hinglish"],
        horizontal=True,
        help=(
            "English translates the call into English. "
            "Hindi gives Devanagari Hindi script. "
            "Hinglish gives Hindi written using English letters."
        )
    )

    st.caption(
        "English = English transcript | Hindi = हिंदी लिपि | Hinglish = Roman Hindi source-style transcript"
    )

    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_audio:
        st.audio(uploaded_audio)

    analyze_clicked = st.button(
        "🚀 Analyze Call",
        type="primary",
        use_container_width=True
    )


if uploaded_audio and analyze_clicked:
    with st.spinner("Transcribing and evaluating the call..."):
        transcript = transcribe_audio(uploaded_audio, transcription_mode)

        if transcript:

            if transcription_mode == "Hinglish":
                fields = extract_fields(transcript)

                bot_rating, bot_score, bot_reason, bot_checks = evaluate_bot(transcript, fields)
                user_rating, user_score, user_reason, user_sentiment, user_checks = evaluate_user(transcript, fields)

                status = call_status(fields)
                summary = create_summary(status, bot_rating, user_rating, fields)

                evaluation_source = "Whisper + Local Evaluation"

            else:
                ai_result = gemini_evaluate_call(transcript, transcription_mode)

                fields = ai_result["fields_confirmed"]

                bot_rating = ai_result["bot_quality"]["rating"]
                bot_reason = ai_result["bot_quality"]["reason"]
                bot_checks = ai_result["bot_quality"]["parameters"]

                user_rating = ai_result["user_quality"]["rating"]
                user_reason = ai_result["user_quality"]["reason"]
                user_sentiment = ai_result["user_quality"]["sentiment"]
                user_checks = ai_result["user_quality"]["parameters"]

                status = ai_result["overall_call_status"]
                summary = ai_result["summary"]

                bot_score = 0
                user_score = 0

                evaluation_source = "Whisper + Gemini AI Evaluation"

            completion, confirmed_count = calculate_completion(fields)

            st.session_state["result"] = {
                "transcript": transcript,
                "transcription_mode": transcription_mode,
                "fields": fields,
                "bot_rating": bot_rating,
                "bot_score": bot_score,
                "bot_reason": bot_reason,
                "bot_checks": bot_checks,
                "user_rating": user_rating,
                "user_score": user_score,
                "user_reason": user_reason,
                "user_sentiment": user_sentiment,
                "user_checks": user_checks,
                "status": status,
                "summary": summary,
                "completion": completion,
                "confirmed_count": confirmed_count,
                "evaluation_source": evaluation_source
            }


if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("## 📊 Evaluation Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Call Status", result["status"])
    col2.metric("Fields Confirmed", f"{result['confirmed_count']}/4")
    col3.metric("Data Completion", f"{result['completion']}%")
    col4.metric(
        "Evaluation Source",
        "Gemini" if "Gemini" in result["evaluation_source"] else "Local"
    )

    st.progress(result["completion"] / 100)
    display_status_badge(result["status"])

    st.divider()

    st.markdown(f"## 1. {result['transcription_mode']} Transcription")
    st.text_area(
        "Generated Transcript",
        result["transcript"],
        height=220
    )

    st.divider()

    st.markdown("## 2. Fields Confirmed")
    fields_df = pd.DataFrame(result["fields"])

    st.dataframe(
        fields_df.style.applymap(status_style, subset=["Status"]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("## 3. Overall Call Quality")

    bot_col, user_col = st.columns(2)

    with bot_col:
        with st.container(border=True):
            st.markdown("### 🤖 A. Bot Quality")
            st.metric("Bot Quality", result["bot_rating"])
            st.write(result["bot_reason"])

            with st.expander("View Bot Quality Parameters"):
                st.dataframe(
                    pd.DataFrame(
                        result["bot_checks"],
                        columns=["Evaluation Parameter", "Result"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    with user_col:
        with st.container(border=True):
            st.markdown("### 👤 B. User Quality")
            st.metric("User Quality", result["user_rating"])
            st.write(result["user_reason"])
            st.write(f"Sentiment: **{result['user_sentiment']}**")

            with st.expander("View User Quality Parameters"):
                st.dataframe(
                    pd.DataFrame(
                        result["user_checks"],
                        columns=["Evaluation Parameter", "Result"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    st.divider()

    st.markdown("## 4. Call Summary")
    st.info(result["summary"])

    st.divider()

    final_json = {
        "transcription_format": result["transcription_mode"],
        "evaluation_source": result["evaluation_source"],
        "transcription": result["transcript"],
        "fields_confirmed": result["fields"],
        "bot_quality": {
            "rating": result["bot_rating"],
            "reason": result["bot_reason"],
            "parameters": result["bot_checks"]
        },
        "user_quality": {
            "rating": result["user_rating"],
            "sentiment": result["user_sentiment"],
            "reason": result["user_reason"],
            "parameters": result["user_checks"]
        },
        "overall_call_status": result["status"],
        "summary": result["summary"]
    }

    st.markdown("## 5. Download Reports")
    st.caption("Downloadable reports can be used for QA review, auditing, and analysis.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "⬇️ Download JSON Report",
            data=json.dumps(final_json, indent=4, ensure_ascii=False),
            file_name="call_evaluation_report.json",
            mime="application/json",
            use_container_width=True
        )

    with col_b:
        st.download_button(
            "⬇️ Download Field Extraction CSV",
            data=fields_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="fields_confirmed.csv",
            mime="text/csv",
            use_container_width=True
        )
