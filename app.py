"""
TrustGuard SA - WhatsApp AI Scam Detection Backend
Detects: Deepfake Images + AI Voice Notes
Platform: Flask + Render + Twilio + Hugging Face
"""

import os
import sys
import time
import requests
import threading
import signal
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
from datetime import date
from collections import defaultdict

app = Flask(__name__)

# ── Environment Variables ──────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN  = os.environ.get('TWILIO_AUTH_TOKEN')
HF_TOKEN           = os.environ.get('HF_TOKEN')
TWILIO_PHONE       = os.environ.get('TWILIO_PHONE', '+15556403201')

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

# ── Thread tracking (prevent garbage collection) ──────────────────
active_threads = {}

# ── Twilio Client ─────────────────────────────────────────────────
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ── Limit Helpers ─────────────────────────────────────────────────
def can_detect(phone):
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        u['count'] = 0
        u['date']  = today
    return u['count'] < DAILY_LIMIT

def use_one(phone):
    usage_tracker[phone]['count'] += 1

def left_today(phone):
    today = date.today()
    u = usage_tracker[phone]
    if u['date'] != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - u['count'])

# ── HF Inference ──────────────────────────────────────────────────
def hf_query(model_url, data):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    print(f"[HF] Query: {model_url}")
    sys.stdout.flush()

    for attempt in range(10):
        try:
            print(f"[HF] Attempt {attempt+1}/10...")
            sys.stdout.flush()
            r = requests.post(model_url, headers=headers, data=data, timeout=120)
            print(f"[HF] Status: {r.status_code} | {r.text[:200]}")
            sys.stdout.flush()

            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                wait = 20 * (attempt + 1)  # 20, 40, 60, 80...
                print(f"[HF] Cold start, wait {wait}s")
                sys.stdout.flush()
                time.sleep(wait)
            else:
                # Give it more chances
                time.sleep(15)
        except Exception as e:
            print(f"[HF] Error: {e}")
            sys.stdout.flush()
            time.sleep(10)

    return None

def top_result(results):
    if not isinstance(results, list) or not results:
        return None, None
    try:
        best = max(results, key=lambda x: x.get('score', 0))
        return best.get('label', '').lower(), round(best.get('score', 0) * 100, 1)
    except:
        return None, None

def is_fake(label):
    label = label.lower()
    if any(w in label for w in ('bonafide', 'real', 'genuine', 'authentic')):
        return False
    return any(w in label for w in ('fake', 'spoof', 'deepfake', 'synthetic', 'generated'))

def fetch_media(url):
    print(f"[MEDIA] Downloading...")
    sys.stdout.flush()
    r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
    r.raise_for_status()
    print(f"[MEDIA] Got {len(r.content)} bytes")
    sys.stdout.flush()
    return r.content

# ── Twilio Send ───────────────────────────────────────────────────
def send_whatsapp(to, body):
    if not twilio_client:
        print("[TWILIO] No client!")
        return False
    try:
        phone = to.replace('whatsapp:', '')
        if not phone.startswith('whatsapp:'):
            phone = f"whatsapp:{phone}"
        msg = twilio_client.messages.create(
            from_=f"whatsapp:{TWILIO_PHONE}",
            body=body,
            to=phone
        )
        print(f"[TWILIO] Sent! {msg.sid}")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[TWILIO] Failed: {e}")
        sys.stdout.flush()
        return False

# ── Templates ─────────────────────────────────────────────────────
WELCOME = ("👋 *Welcome to TrustGuard SA!*\n"
           "South Africa's #1 AI scam detection service 🛡️\n\n"
           "Here's what I can do for you:\n"
           "📸 Send me an *image* — I'll check if it's a deepfake\n"
           "🎵 Send me a *voice note* — I'll check if it's AI generated\n\n"
           "🔜 *Coming soon:*\n✨ Video detection\n✨ Document verification\n✨ Live scam call alerts\n\n"
           "Send me anything suspicious! 🛡️")

UNKNOWN = ("🛡️ I'm TrustGuard SA — I only detect deepfakes and AI voices.\n"
           "Send me an *image* 📸 or *voice note* 🎵 to get started!")

LIMIT_MSG = ("⚠️ You've used all *3 daily detections*.\n"
             "Come back tomorrow for more protection!\n\n"
             "🔜 *Premium plan coming soon* for unlimited access.")

ERROR_MSG = ("⚠️ Something went wrong during analysis.\n"
             "Please try again in a moment.")

ANALYZING_IMAGE = ("🔍 *Analyzing your image...*\n\n"
                   "Our AI is checking for deepfakes. This may take 30-60 seconds.\n"
                   "I'll send you the results as soon as they're ready!")

ANALYZING_VOICE = ("🔍 *Analyzing your voice note...*\n\n"
                   "Our AI is checking for voice clones. This may take 30-60 seconds.\n"
                   "I'll send you the results as soon as they're ready!")

def closing(left):
    return (f"Protect yourself and your family always! 🛡️\n\n"
            f"You have *{left} detection(s)* left for today.\n\n"
            "📸 Send an image or 🎵 voice note anytime you feel suspicious.\n\n"
            "🔜 *More features coming soon!*")

def image_reply(label, conf, left):
    if is_fake(label):
        v = ("⚠️ *Deepfake Detected!*\n"
             f"This image appears to be AI generated — *{conf}% confidence*\n"
             "🚨 Do not trust this image!")
    else:
        v = ("✅ *Image Looks Real*\n"
             f"This image appears to be genuine — *{conf}% confidence*")
    return f"{v}\n\n{closing(left)}"

def voice_reply(label, conf, left):
    if is_fake(label):
        v = ("⚠️ *AI Voice Detected!*\n"
             f"This voice note appears to be AI generated — *{conf}% confidence*\n"
             "🚨 Do not trust this voice!")
    else:
        v = ("✅ *Voice Sounds Real*\n"
             f"This voice note appears to be genuine — *{conf}% confidence*")
    return f"{v}\n\n{closing(left)}"

# ── Background Worker ─────────────────────────────────────────────
def run_analysis(sender, media_url, content_type):
    """Background thread: download, analyze, send result."""
    global active_threads
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Started for {sender}")
    sys.stdout.flush()

    try:
        # Download
        try:
            data = fetch_media(media_url)
        except Exception as e:
            print(f"[{thread_name}] Download failed: {e}")
            send_whatsapp(sender, ERROR_MSG)
            return

        # Pick model
        if content_type.startswith('image/'):
            model_url, reply_fn = IMAGE_MODEL, image_reply
        elif content_type.startswith('audio/'):
            model_url, reply_fn = VOICE_MODEL, voice_reply
        else:
            send_whatsapp(sender, UNKNOWN)
            return

        # HF Inference with extended retries
        results = hf_query(model_url, data)
        label, conf = top_result(results)

        if label:
            reply_text = reply_fn(label, conf, left_today(sender))
        else:
            reply_text = ERROR_MSG

        send_whatsapp(sender, reply_text)
        print(f"[{thread_name}] Done!")
        sys.stdout.flush()

    except Exception as e:
        print(f"[{thread_name}] Error: {e}")
        sys.stdout.flush()
        send_whatsapp(sender, ERROR_MSG)
    finally:
        active_threads.pop(thread_name, None)

# ── Keep-Alive (pings itself to stay warm) ────────────────────────
def keep_alive():
    """Ping the service every 10 minutes to prevent Render from spinning down."""
    while True:
        try:
            time.sleep(600)  # 10 minutes
            url = os.environ.get('RENDER_EXTERNAL_URL', 'https://trustguard-sa.onrender.com')
            requests.get(f"{url}/", timeout=10)
            print("[KEEPALIVE] Pinged self")
        except:
            pass

# Start keep-alive thread
ka_thread = threading.Thread(target=keep_alive, daemon=True, name="keepalive")
ka_thread.start()

# ── Endpoints ─────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    threads = list(active_threads.keys())
    return (f"🛡️ TrustGuard SA Backend is live!\n"
            f"Active threads: {threads}"), 200

@app.route('/webhook', methods=['POST'])
def webhook():
    global seen_users, active_threads

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
        elif content_type.startswith('image/') or content_type.startswith('audio/'):
            use_one(sender)
            media_type = 'image' if content_type.startswith('image/') else 'audio'
            reply = ANALYZING_IMAGE if media_type == 'image' else ANALYZING_VOICE

            # Launch background thread (daemon=False to keep alive after request)
            thread_name = f"worker_{sender.replace('+','')}_{int(time.time())}"
            t = threading.Thread(
                target=run_analysis,
                args=(sender, media_url, content_type),
                name=thread_name,
                daemon=False  # Non-daemon: thread keeps running after request ends
            )
            active_threads[thread_name] = t
            t.start()
        else:
            reply = UNKNOWN

    elif body in GREETINGS or any(g in body for g in GREETINGS):
        reply = WELCOME
    else:
        reply = UNKNOWN

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {'Content-Type': 'text/xml'}

# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[STARTUP] TrustGuard SA starting on port {port}")
    sys.stdout.flush()
    app.run(host='0.0.0.0', port=port, debug=False)
