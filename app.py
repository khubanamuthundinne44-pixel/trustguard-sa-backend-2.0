"""
TrustGuard SA - WhatsApp AI Scam Detection Backend
Detects: Deepfake Images + AI Voice Notes
Platform: Flask + Render + Twilio + Hugging Face
"""

import os
import sys
import time
import json
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
TWILIO_PHONE       = os.environ.get('TWILIO_PHONE', '+155****3201')

# ── Hugging Face AI Models ────────────────────────────────────────
# Use router.huggingface.co instead of api-inference.huggingface.co
# because api-inference.huggingface.co has no public DNS records
# and cannot be resolved from Render's network
IMAGE_MODEL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"
VOICE_MODEL = "https://router.huggingface.co/hf-inference/models/garystafford/wav2vec2-deepfake-voice-detector"

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

# ── Debug Store (for troubleshooting without Render logs) ─────────
debug_logs = []          # list of dicts with debug info
debug_lock = threading.Lock()

def add_debug(entry):
    """Store debug entry with timestamp. Keep last 50 entries."""
    entry['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with debug_lock:
        debug_logs.append(entry)
        if len(debug_logs) > 50:
            debug_logs.pop(0)
    # Also write to file for persistence across restarts
    try:
        with open('/tmp/trustguard_debug.json', 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except:
        pass

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
def hf_query(model_url, data, task="image-classification"):
    """Query HuggingFace Inference API via router.huggingface.co"""
    model_name = model_url.split('/models/')[-1]
    add_debug({"event": "hf_start", "model": model_name, "task": task})

    # Try raw requests first with proper content-type header
    # (InferenceClient sometimes fails with content-type issues on router)
    content_type = "audio/wav" if task == "audio-classification" else "image/png"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": content_type
    }

    for attempt in range(5):
        try:
            r = requests.post(model_url, headers=headers, data=data, timeout=120)
            add_debug({
                "event": "hf_attempt",
                "attempt": attempt + 1,
                "status": r.status_code,
                "response": r.text[:500]
            })

            if r.status_code == 200:
                return r.json()
            if r.status_code == 503:
                # Model is loading, wait and retry
                result = r.json() if r.text else {}
                estimated = result.get('estimated_time', 30)
                wait = min(estimated + 5, 120)
                add_debug({"event": "hf_cold_start", "wait": wait})
                time.sleep(wait)
            elif r.status_code == 401:
                add_debug({"event": "hf_auth_error"})
                return None
            else:
                time.sleep(15)
        except Exception as e:
            add_debug({"event": "hf_error", "attempt": attempt + 1, "error": str(e)})
            time.sleep(10)

    add_debug({"event": "hf_all_failed"})
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
    add_debug({"event": "media_download", "url": url})
    r = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
    r.raise_for_status()
    add_debug({"event": "media_downloaded", "size": len(r.content)})
    return r.content

# ── Twilio Send ───────────────────────────────────────────────────
def send_whatsapp(to, body):
    if not twilio_client:
        add_debug({"event": "twilio_error", "error": "No client configured"})
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
        add_debug({"event": "twilio_sent", "sid": msg.sid, "to": phone})
        return True
    except Exception as e:
        add_debug({"event": "twilio_send_error", "error": str(e)})
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
    add_debug({"event": "thread_start", "thread": thread_name, "sender": sender, "type": content_type})

    try:
        # Download
        try:
            data = fetch_media(media_url)
        except Exception as e:
            add_debug({"event": "media_error", "error": str(e), "thread": thread_name})
            send_whatsapp(sender, ERROR_MSG)
            return

        # Pick model
        if content_type.startswith('image/'):
            model_url, reply_fn, task = IMAGE_MODEL, image_reply, "image-classification"
        elif content_type.startswith('audio/'):
            model_url, reply_fn, task = VOICE_MODEL, voice_reply, "audio-classification"
        else:
            add_debug({"event": "unknown_type", "content_type": content_type})
            send_whatsapp(sender, UNKNOWN)
            return

        # HF Inference
        results = hf_query(model_url, data, task=task)
        label, conf = top_result(results)

        add_debug({
            "event": "analysis_result",
            "thread": thread_name,
            "label": label,
            "conf": conf,
            "raw_results": str(results)[:500] if results else None
        })

        if label:
            reply_text = reply_fn(label, conf, left_today(sender))
        else:
            reply_text = ERROR_MSG

        sent = send_whatsapp(sender, reply_text)
        add_debug({"event": "thread_done", "thread": thread_name, "sent": sent})

    except Exception as e:
        add_debug({"event": "thread_error", "thread": thread_name, "error": str(e)})
        send_whatsapp(sender, ERROR_MSG)
    finally:
        active_threads.pop(thread_name, None)

# ── Keep-Alive ────────────────────────────────────────────────────
def keep_alive():
    while True:
        try:
            time.sleep(600)
            url = os.environ.get('RENDER_EXTERNAL_URL', 'https://trustguard-sa.onrender.com')
            requests.get(f"{url}/", timeout=10)
        except:
            pass

ka_thread = threading.Thread(target=keep_alive, daemon=True, name="keepalive")
ka_thread.start()

# ── Endpoints ─────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    threads = list(active_threads.keys())
    return (f"🛡️ TrustGuard SA Backend is live!\n"
            f"Active threads: {threads}"), 200

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint: shows recent debug logs."""
    with debug_lock:
        logs = list(debug_logs)
    # Also try to read from file
    file_logs = []
    try:
        with open('/tmp/trustguard_debug.json', 'r') as f:
            for line in f:
                try:
                    file_logs.append(json.loads(line.strip()))
                except:
                    pass
    except:
        pass
    return json.dumps({"memory_logs": logs, "file_logs": file_logs[-30:]}, indent=2, default=str), 200

@app.route('/test', methods=['GET'])
def test_connectivity():
    """Test outbound connectivity from the Render server."""
    import socket
    results = {}

    # Test DNS resolution
    for host in ['google.com', 'huggingface.co', 'api-inference.huggingface.co', 'httpbin.org']:
        try:
            ip = socket.gethostbyname(host)
            results[f'dns_{host}'] = f'OK ({ip})'
        except Exception as e:
            results[f'dns_{host}'] = f'FAIL: {e}'

    # Test HTTP connectivity
    for name, url in [
        ('google', 'https://www.google.com'),
        ('hf', 'https://huggingface.co'),
        ('hf_api', 'https://api-inference.huggingface.co'),
        ('httpbin', 'https://httpbin.org/get'),
    ]:
        try:
            r = requests.get(url, timeout=10)
            results[f'http_{name}'] = f'OK ({r.status_code})'
        except Exception as e:
            results[f'http_{name}'] = f'FAIL: {e}'

    # Test if HF token is set
    results['hf_token_set'] = bool(HF_TOKEN)
    results['hf_token_prefix'] = HF_TOKEN[:10] + '...' if HF_TOKEN else 'NOT SET'

    # Test Twilio credentials
    results['twilio_sid_set'] = bool(TWILIO_ACCOUNT_SID)
    results['twilio_token_set'] = bool(TWILIO_AUTH_TOKEN)
    results['twilio_phone'] = TWILIO_PHONE

    return json.dumps(results, indent=2), 200

@app.route('/webhook', methods=['POST'])
def webhook():
    global seen_users, active_threads

    sender    = request.form.get('From', '')
    body      = request.form.get('Body', '').strip().lower()
    num_media = int(request.form.get('NumMedia', 0))

    add_debug({"event": "webhook", "sender": sender, "body": body, "num_media": num_media})

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

            thread_name = f"worker_{sender.replace('+','')}_{int(time.time())}"
            t = threading.Thread(
                target=run_analysis,
                args=(sender, media_url, content_type),
                name=thread_name,
                daemon=False
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
    add_debug({"event": "startup", "port": port})
    app.run(host='0.0.0.0', port=port, debug=False)
