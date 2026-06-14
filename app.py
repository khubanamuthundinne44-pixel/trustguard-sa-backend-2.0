"""
TrustGuard SA - WhatsApp AI Scam Detection Backend
Detects: Deepfake Images + AI Voice Notes
Platform: Flask + Render + Twilio + Hugging Face
"""

import os
import time
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from datetime import date
from collections import defaultdict

app = Flask(__name__)

# -- Environment Variables (set these on Render) --
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN  = os.environ.get('TWILIO_AUTH_TOKEN')
HF_TOKEN           = os.environ.get('HF_TOKEN')

# -- Hugging Face AI Models --
IMAGE_MODEL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
VOICE_MODEL = "https://api-inference.huggingface.co/models/garystafford/wav2vec2-deepfake-voice-detector"

# -- Settings --
DAILY_LIMIT = 3

GREETINGS = {
    'hi', 'hello', 'hey', 'start', 'hola', 'howzit', 'sup', 'yo',
    'greetings', 'good morning', 'good afternoon', 'good evening',
    'morning', 'evening', 'afternoon', 'heita', 'sawubona', 'dumela',
    'sanibonani', 'molo', 'avuxeni', 'ndaa', 'ndi madekwana'
}

# -- In-Memory State --
usage_tracker = defaultdict(lambda: {'count': 0, 'date': None})
seen_users    = set()

# -- Limit Helpers --
def can_detect(phone: str) -> bool:
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        u['count'] = 0
        u['date']  = today
    return u['count'] < DAILY_LIMIT

def use_one(phone: str):
    usage_tracker[phone]['count'] += 1

def left_today(phone: str) -> int:
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - u['count'])

# -- Hugging Face Inference --
def hf_query(model_url: str, data: bytes):
    """Send file to HF model. Handles cold-start by waiting for model to load."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # First request -- may return 503 if model is cold
    try:
        r = requests.post(model_url, headers=headers, data=data, timeout=60)
    except requests.RequestException:
        return None

    # If model is loading, wait and retry
    if r.status_code == 503:
        try:
            estimated = r.json().get("estimated_time", 30)
        except Exception:
            estimated = 30
        wait_time = min(int(estimated) + 5, 90)
        time.sleep(wait_time)
        try:
            r = requests.post(model_url, headers=headers, data=data, timeout=60)
        except requests.RequestException:
            return None

    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return None

    return None

def top_result(results):
    if not isinstance(results, list) or not results:
        return None, None
    best = max(results, key=lambda x: x.get('score', 0))
    return best.get('label', '').lower(), round(best.get('score', 0) * 100, 1)

# -- Label Classification --
def is_fake(label: str) -> bool:
    label = label.lower()
    if any(w in label for w in ('bonafide', 'real', 'genuine', 'authentic')):
        return False
    return any(w in label for w in ('fake', 'spoof', 'deepfake', 'synthetic', 'generated'))

# -- Download Media from Twilio --
def fetch_media(url: str) -> bytes:
    r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
    r.raise_for_status()
    return r.content

# -- Message Templates --
WELCOME = (
    "\U0001f44b *Welcome to TrustGuard SA!*\n"
    "South Africa's #1 AI scam detection service \U0001f6e1\ufe0f\n\n"
    "Here's what I can do for you:\n"
    "\U0001f4f8 Send me an *image* \u2014 I'll check if it's a deepfake\n"
    "\U0001f3b5 Send me a *voice note* \u2014 I'll check if it's AI generated\n\n"
    "\U0001f51c *Coming soon:*\n"
    "\u2728 Video detection\n"
    "\u2728 Document verification\n"
    "\u2728 Live scam call alerts\n\n"
    "Send me anything suspicious! \U0001f6e1\ufe0f"
)

UNKNOWN = (
    "\U0001f6e1\ufe0f I'm TrustGuard SA \u2014 I only detect deepfakes and AI voices.\n"
    "Send me an *image* \U0001f4f8 or *voice note* \U0001f3b5 to get started!"
)

LIMIT_MSG = (
    "\u26a0\ufe0f You've used all *3 daily detections*.\n"
    "Come back tomorrow for more protection!\n\n"
    "\U0001f51c *Premium plan coming soon* for unlimited access."
)

ERROR_MSG = (
    "\u26a0\ufe0f Something went wrong during analysis.\n"
    "Please try again in a moment."
)

MODEL_DOWN_MSG = (
    "\u26a0\ufe0f The AI model is currently overloaded.\n"
    "Please try again in 30 seconds."
)

def closing(left: int) -> str:
    return (
        f"Protect yourself and your family always! \U0001f6e1\ufe0f\n\n"
        f"You have *{left} detection(s)* left for today.\n\n"
        "\U0001f4f8 Send an image or \U0001f3b5 voice note anytime you feel suspicious.\n\n"
        "\U0001f51c *More features coming soon!*"
    )

def image_reply(label: str, conf: float, left: int) -> str:
    if is_fake(label):
        verdict = (
            "\u26a0\ufe0f *Deepfake Detected!*\n"
            f"This image appears to be AI generated \u2014 *{conf}% confidence*\n"
            "\U0001f6a8 Do not trust this image!"
        )
    else:
        verdict = (
            "\u2705 *Image Looks Real*\n"
            f"This image appears to be genuine \u2014 *{conf}% confidence*"
        )
    return f"{verdict}\n\n{closing(left)}"

def voice_reply(label: str, conf: float, left: int) -> str:
    if is_fake(label):
        verdict = (
            "\u26a0\ufe0f *AI Voice Detected!*\n"
            f"This voice note appears to be AI generated \u2014 *{conf}% confidence*\n"
            "\U0001f6a8 Do not trust this voice!"
        )
    else:
        verdict = (
            "\u2705 *Voice Sounds Real*\n"
            f"This voice note appears to be genuine \u2014 *{conf}% confidence*"
        )
    return f"{verdict}\n\n{closing(left)}"

# -- Main Webhook --
@app.route('/webhook', methods=['POST'])
def webhook():
    sender    = request.form.get('From', '')
    body      = request.form.get('Body', '').strip().lower()
    num_media = int(request.form.get('NumMedia', 0))

    reply = UNKNOWN

    if sender not in seen_users:
        seen_users.add(sender)
        reply = WELCOME

    elif num_media > 0:
        media_url    = request.form.get('MediaUrl0', '')
        content_type = request.form.get('MediaContentType0', '')

        if not can_detect(sender):
            reply = LIMIT_MSG

        elif content_type.startswith('image/'):
            try:
                data = fetch_media(media_url)
            except Exception:
                reply = ERROR_MSG
            else:
                use_one(sender)
                results = hf_query(IMAGE_MODEL, data)
                if results is None:
                    reply = MODEL_DOWN_MSG
                else:
                    label, conf = top_result(results)
                    reply = image_reply(label, conf, left_today(sender)) if label else ERROR_MSG

        elif content_type.startswith('audio/'):
            try:
                data = fetch_media(media_url)
            except Exception:
                reply = ERROR_MSG
            else:
                use_one(sender)
                results = hf_query(VOICE_MODEL, data)
                if results is None:
                    reply = MODEL_DOWN_MSG
                else:
                    label, conf = top_result(results)
                    reply = voice_reply(label, conf, left_today(sender)) if label else ERROR_MSG

        else:
            reply = UNKNOWN

    elif body in GREETINGS or any(g in body for g in GREETINGS):
        reply = WELCOME

    else:
        reply = UNKNOWN

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {'Content-Type': 'text/xml'}

# -- Health Check --
@app.route('/', methods=['GET'])
def home():
    return "\U0001f6e1\ufe0f TrustGuard SA Backend is live!", 200

# -- Run --
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
