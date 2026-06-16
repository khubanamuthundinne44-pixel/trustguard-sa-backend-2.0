"""
TrustGuard SA - WhatsApp AI Scam Detection Backend
Detects: Deepfake Images + AI Voice Notes
Platform: Flask + Render + Twilio + Hugging Face
"""

import os
import time
import requests
import threading
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from datetime import date
from collections import defaultdict

app = Flask(__name__)

# ── Environment Variables (set these on Render) ───────────────────
# TWILIO_ACCOUNT_SID  → from twilio.com/console
# TWILIO_AUTH_TOKEN   → from twilio.com/console
# HF_TOKEN            → from huggingface.co/settings/tokens

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN  = os.environ.get('TWILIO_AUTH_TOKEN')
HF_TOKEN           = os.environ.get('HF_TOKEN')

# ── Hugging Face AI Models ────────────────────────────────────────
IMAGE_MODEL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
VOICE_MODEL = "https://api-inference.huggingface.co/models/garystafford/wav2vec2-deepfake-voice-detector"

# ── Settings ──────────────────────────────────────────────────────
DAILY_LIMIT = 3

GREETINGS = {
    'hi', 'hello', 'hey', 'start', 'hola', 'howzit', 'sup', 'yo',
    'greetings', 'good morning', 'good afternoon', 'good evening',
    'morning', 'evening', 'afternoon', 'heita', 'sawubona', 'dumela',
    'sanibonani', 'molo', 'avuxeni', 'ndaa', 'ndi madekwana'
}

# ── In-Memory State ───────────────────────────────────────────────
usage_tracker = defaultdict(lambda: {'count': 0, 'date': None})
seen_users    = set()

# ── Twilio Client (for sending async replies) ─────────────────────
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ── Limit Helpers ─────────────────────────────────────────────────
def can_detect(phone: str) -> bool:
    """Check if user still has detections left today."""
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        u['count'] = 0
        u['date']  = today
    return u['count'] < DAILY_LIMIT

def use_one(phone: str):
    """Consume one detection from user's daily limit."""
    usage_tracker[phone]['count'] += 1

def left_today(phone: str) -> int:
    """Return how many detections user has left today."""
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - u['count'])

# ── Hugging Face Inference ────────────────────────────────────────
def hf_query(model_url: str, data: bytes):
    """Send file to HF model and return results. Retries if model is loading."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    print(f"[HF] Querying model: {model_url}")
    print(f"[HF] Token set: {'Yes' if HF_TOKEN else 'NO TOKEN FOUND!'}")
    print(f"[HF] Data size: {len(data)} bytes")

    for attempt in range(5):
        try:
            print(f"[HF] Attempt {attempt + 1}...")
            r = requests.post(model_url, headers=headers, data=data, timeout=120)
            print(f"[HF] Status code: {r.status_code}")
            print(f"[HF] Response: {r.text[:300]}")

            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                wait = 15
                print(f"[HF] Model loading, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[HF] Unexpected status: {r.status_code}")
                break
        except requests.RequestException as e:
            print(f"[HF] Request error: {e}")
            time.sleep(5)

    print("[HF] All attempts failed, returning None")
    return None

def top_result(results):
    """Extract the highest-confidence label from HF classifier output."""
    if not isinstance(results, list) or not results:
        return None, None
    best = max(results, key=lambda x: x.get('score', 0))
    return best.get('label', '').lower(), round(best.get('score', 0) * 100, 1)

# ── Label Classification ──────────────────────────────────────────
def is_fake(label: str) -> bool:
    """Return True if the HF label means fake/AI-generated."""
    label = label.lower()
    # These labels mean REAL
    if any(w in label for w in ('bonafide', 'real', 'genuine', 'authentic')):
        return False
    # These labels mean FAKE
    return any(w in label for w in ('fake', 'spoof', 'deepfake', 'synthetic', 'generated'))

# ── Download Media from Twilio ────────────────────────────────────
def fetch_media(url: str) -> bytes:
    """Download media file sent via WhatsApp (requires Twilio auth)."""
    print(f"[MEDIA] Downloading: {url}")
    r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
    print(f"[MEDIA] Status: {r.status_code}")
    r.raise_for_status()
    print(f"[MEDIA] Success! Got {len(r.content)} bytes")
    return r.content

# ── Message Templates ─────────────────────────────────────────────
WELCOME = (
    "👋 *Welcome to TrustGuard SA!*\n"
    "South Africa's #1 AI scam detection service 🛡️\n\n"
    "Here's what I can do for you:\n"
    "📸 Send me an *image* — I'll check if it's a deepfake\n"
    "🎵 Send me a *voice note* — I'll check if it's AI generated\n\n"
    "🔜 *Coming soon:*\n"
    "✨ Video detection\n"
    "✨ Document verification\n"
    "✨ Live scam call alerts\n\n"
    "Send me anything suspicious! 🛡️"
)

UNKNOWN = (
    "🛡️ I'm TrustGuard SA — I only detect deepfakes and AI voices.\n"
    "Send me an *image* 📸 or *voice note* 🎵 to get started!"
)

LIMIT_MSG = (
    "⚠️ You've used all *3 daily detections*.\n"
    "Come back tomorrow for more protection!\n\n"
    "🔜 *Premium plan coming soon* for unlimited access."
)

ERROR_MSG = (
    "⚠️ Something went wrong during analysis.\n"
    "Please try again in a moment."
)

ANALYZING_IMAGE = (
    "🔍 *Analyzing your image...*\n\n"
    "Our AI is checking for deepfakes. This may take 30-60 seconds.\n"
    "I'll send you the results as soon as they're ready!"
)

ANALYZING_VOICE = (
    "🔍 *Analyzing your voice note...*\n\n"
    "Our AI is checking for voice clones. This may take 30-60 seconds.\n"
    "I'll send you the results as soon as they're ready!"
)

def closing(left: int) -> str:
    return (
        f"Protect yourself and your family always! 🛡️\n\n"
        f"You have *{left} detection(s)* left for today.\n\n"
        "📸 Send an image or 🎵 voice note anytime you feel suspicious.\n\n"
        "🔜 *More features coming soon!*"
    )

def image_reply(label: str, conf: float, left: int) -> str:
    if is_fake(label):
        verdict = (
            "⚠️ *Deepfake Detected!*\n"
            f"This image appears to be AI generated — *{conf}% confidence*\n"
            "🚨 Do not trust this image!"
        )
    else:
        verdict = (
            "✅ *Image Looks Real*\n"
            f"This image appears to be genuine — *{conf}% confidence*"
        )
    return f"{verdict}\n\n{closing(left)}"

def voice_reply(label: str, conf: float, left: int) -> str:
    if is_fake(label):
        verdict = (
            "⚠️ *AI Voice Detected!*\n"
            f"This voice note appears to be AI generated — *{conf}% confidence*\n"
            "🚨 Do not trust this voice!"
        )
    else:
        verdict = (
            "✅ *Voice Sounds Real*\n"
            f"This voice note appears to be genuine — *{conf}% confidence*"
        )
    return f"{verdict}\n\n{closing(left)}"

# ── Async Analysis (Background Thread) ────────────────────────────
def async_analyze(sender: str, media_url: str, content_type: str):
    """Run HF analysis in background and send result via Twilio REST API."""
    print(f"[ASYNC] Starting analysis for {sender}")
    print(f"[ASYNC] Content type: {content_type}")

    try:
        # Download media
        try:
            data = fetch_media(media_url)
        except Exception as e:
            print(f"[ASYNC] Media download failed: {e}")
            send_whatsapp(sender, ERROR_MSG)
            return

        # Pick model
        if content_type.startswith('image/'):
            model_url = IMAGE_MODEL
            reply_fn  = image_reply
        elif content_type.startswith('audio/'):
            model_url = VOICE_MODEL
            reply_fn  = voice_reply
        else:
            send_whatsapp(sender, UNKNOWN)
            return

        # Query HF (this can take a while)
        results = hf_query(model_url, data)
        label, conf = top_result(results)

        if label:
            left = left_today(sender)
            reply = reply_fn(label, conf, left)
        else:
            reply = ERROR_MSG

        # Send result back via Twilio REST API
        send_whatsapp(sender, reply)
        print(f"[ASYNC] Result sent to {sender}")

    except Exception as e:
        print(f"[ASYNC] Unexpected error: {e}")
        send_whatsapp(sender, ERROR_MSG)

def send_whatsapp(to: str, body: str):
    """Send a WhatsApp message via Twilio REST API."""
    if not twilio_client:
        print("[TWILIO] Client not configured, cannot send message")
        return
    try:
        # Extract the phone number without 'whatsapp:' prefix if present
        phone = to.replace('whatsapp:', '')
        # Ensure it has whatsapp: prefix for Twilio
        if not phone.startswith('whatsapp:'):
            phone = f"whatsapp:{phone}"

        msg = twilio_client.messages.create(
            from_=f"whatsapp:{os.environ.get('TWILIO_PHONE', '+15556403201')}",
            body=body,
            to=phone
        )
        print(f"[TWILIO] Message sent! SID: {msg.sid}")
    except Exception as e:
        print(f"[TWILIO] Failed to send message: {e}")

# ── Main Webhook ──────────────────────────────────────────────────
@app.route('/webhook', methods=['POST'])
def webhook():
    sender    = request.form.get('From', '')
    body      = request.form.get('Body', '').strip().lower()
    num_media = int(request.form.get('NumMedia', 0))

    reply = UNKNOWN  # default fallback

    # ── First-time user → show welcome ────────────────────────────
    if sender not in seen_users:
        seen_users.add(sender)
        reply = WELCOME

    # ── Media sent (image or voice note) ──────────────────────────
    elif num_media > 0:
        media_url    = request.form.get('MediaUrl0', '')
        content_type = request.form.get('MediaContentType0', '')

        if not can_detect(sender):
            reply = LIMIT_MSG

        elif content_type.startswith('image/'):
            use_one(sender)
            # Start background thread for analysis
            thread = threading.Thread(
                target=async_analyze,
                args=(sender, media_url, content_type),
                daemon=True
            )
            thread.start()
            reply = ANALYZING_IMAGE

        elif content_type.startswith('audio/'):
            use_one(sender)
            # Start background thread for analysis
            thread = threading.Thread(
                target=async_analyze,
                args=(sender, media_url, content_type),
                daemon=True
            )
            thread.start()
            reply = ANALYZING_VOICE

        else:
            # Video or unsupported file
            reply = UNKNOWN

    # ── Greeting text ──────────────────────────────────────────────
    elif body in GREETINGS or any(g in body for g in GREETINGS):
        reply = WELCOME

    # ── Anything else (thank you, questions, arguments, etc.) ──────
    else:
        reply = UNKNOWN

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {'Content-Type': 'text/xml'}

# ── Health Check ──────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return "🛡️ TrustGuard SA Backend is live!", 200

# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
