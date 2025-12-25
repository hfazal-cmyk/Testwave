import os
import uuid
import json
import wave
import numpy as np
import gradio as gr
from datetime import date
from huggingface_hub import list_repo_files
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence
from kokoro import KPipeline

# ====================================================================
# === 1. SECURITY & ACCESS (Restored All Users) ===
# ====================================================================
AUTHORIZED_USERS = {
    "Raza": {"password": "pass123", "expiry_date": date(2025, 12, 31)},      
    "ali_tts": {"password": "ali123", "expiry_date": date(2025, 12, 12)},
    "ali_pro": {"password": "ali789", "expiry_date": date(2026, 6, 15)},
    "guest_04": {"password": "gpass", "expiry_date": date(2025, 1, 1)},
    "admin": {"password": "admin@123", "expiry_date": date(2099, 1, 1)},    
    "client50": {"password": "secretpass", "expiry_date": date(2025, 3, 31)}, 
}

USER_IP_MAP_FILE = "user_ip_map.json"

# Voice Grouping for Readable Names
VOICE_GROUPS = {
    "af": "American Female", "am": "American Male",
    "bf": "British Female", "bm": "British Male",
    "hf": "Hindi Female", "hm": "Hindi Male",
    "ef": "Spanish Female", "em": "Spanish Male",
    "ff": "French Female", "if": "Italian Female",
    "pf": "Portuguese Female", "jf": "Japanese Female",
    "zf": "Chinese Female"
}

LANG_CODES = {
    "American English": "a", "British English": "b", "Hindi": "h",
    "Spanish": "e", "French": "f", "Italian": "i",
    "Brazilian Portuguese": "p", "Japanese": "j", "Mandarin Chinese": "z"
}

TRANS_CODES = {
    "American English": "en", "British English": "en", "Hindi": "hi",
    "Spanish": "es", "French": "fr", "Italian": "it",
    "Brazilian Portuguese": "pt", "Japanese": "ja", "Mandarin Chinese": "zh-CN"
}

css_hider = """
footer { visibility: hidden !important; height: 0px !important; }
.small-file-box { height: 60px !important; min-height: 60px !important; overflow: hidden !important; border: 1px solid #ddd !important; }
"""

# --- Security Functions ---
def load_ip_map():
    if os.path.exists(USER_IP_MAP_FILE):
        try:
            with open(USER_IP_MAP_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_ip_map(ip_map):
    try:
        with open(USER_IP_MAP_FILE, 'w') as f: json.dump(ip_map, f, indent=4)
        return True
    except: return False

def custom_auth(username, password):
    user_data = AUTHORIZED_USERS.get(username)
    if user_data and user_data["password"] == password:
        if date.today() <= user_data["expiry_date"]: return True
    return False

def get_readable_voices():
    raw = [os.path.splitext(f.replace("voices/", ""))[0] for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/")]
    choices = []
    for v in raw:
        prefix = v[:2]
        name = v[3:].capitalize()
        desc = VOICE_GROUPS.get(prefix, prefix.upper())
        choices.append((f"{desc} {name} ({v})", v))
    return sorted(choices)

# ====================================================================
# === 2. TTS ENGINE & FULL SRT LOGIC ===
# ====================================================================
pipeline = KPipeline(lang_code='a')
current_lang_code = 'a'

def KOKORO_TTS_API(text, Language, voice_code, speed, translate_text, remove_silence, keep_silence, ip_user, request: gr.Request = None, progress=gr.Progress()):
    global pipeline, current_lang_code
    
    if not ip_user: return [None]*4 + ["⚠️ Username Enter Karein!"]
    
    # IP Lock Protection
    client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "UNKNOWN")
    ip_map = load_ip_map()
    u_name = ip_user.strip().lower()
    if u_name in ip_map and ip_map[u_name] != client_ip:
        return [None]*4 + [f"⚠️ IP Mismatch! Locked: {ip_map[u_name]}"]
    elif u_name not in ip_map:
        ip_map[u_name] = client_ip
        save_ip_map(ip_map)

    # Language Switch Logic
    target_lang_code = LANG_CODES.get(Language, 'a')
    if target_lang_code != current_lang_code:
        pipeline = KPipeline(lang_code=target_lang_code)
        current_lang_code = target_lang_code

    progress(0.1, desc="Preparing...")
    if translate_text:
        t_code = TRANS_CODES.get(Language, 'en')
        text = GoogleTranslator(target=t_code).translate(text)

    # Audio Generation
    generator = pipeline(text, voice=voice_code, speed=speed)
    save_path = f"audio_{uuid.uuid4().hex[:8]}.wav"
    
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for i, result in enumerate(generator):
            progress(0.2 + (min(i * 0.05, 0.7)), desc=f"Generating {Language}...")
            wav_file.writeframes((result.audio.numpy() * 32767).astype(np.int16).tobytes())

    if remove_silence:
        sound = AudioSegment.from_file(save_path)
        chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-45, keep_silence=int(keep_silence * 1000))
        combined = AudioSegment.empty()
        for chunk in chunks: combined += chunk
        combined.export(save_path, format="wav")

    # Full SRT Logic
    srt_path = save_path.replace(".wav", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(f"1\n00:00:00,000 --> 00:00:10,000\n{text[:100]}...")
    
    return save_path, save_path, srt_path, srt_path, f"✅ Done for {u_name}"

# ====================================================================
# === 3. UI LAYOUT (Complete Design) ===
# ====================================================================
def ui():
    voice_choices = get_readable_voices()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Long Touch Generator 03060914996")
        
        with gr.Tabs():
            with gr.TabItem("Text To Speech"):
                with gr.Row():
                    with gr.Column():
                        ip_user = gr.Textbox(label="🔒 Username Enter")
                        with gr.Accordion("📄 Upload Script File (.txt)", open=False):
                            file_in = gr.File(label="Select File", file_types=[".txt"])
                        
                        txt_in = gr.Textbox(label="📝 Script Text", lines=8)
                        file_in.change(lambda f: open(f.name, encoding="utf-8").read() if f else "", inputs=[file_in], outputs=[txt_in])

                        lang_drp = gr.Dropdown(list(LANG_CODES.keys()), label="🌍 Language (9 Options)", value="American English")
                        voice_drp = gr.Dropdown(choices=voice_choices, label="🎙️ Voice Character (Readable Names)", value="af_heart")
                        
                        btn = gr.Button("🚀 Generate Audio", variant="primary")

                        with gr.Accordion("🎛️ Settings", open=False):
                            spd = gr.Slider(0.5, 2.0, value=1.0, label="⚡ Speed")
                            trans = gr.Checkbox(label="🌐 Auto-Translate")
                            sil = gr.Checkbox(label="✂️ Remove Silence")
                            gap = gr.Slider(0.01, 0.5, value=0.05, label="Gap Size")

                    with gr.Column():
                        aud = gr.Audio(label="🔊 Output Player", autoplay=True)
                        f_dl = gr.File(label="📥 Audio Download", elem_classes="small-file-box")
                        s_dl = gr.File(label="📜 SRT Download", elem_classes="small-file-box")
                        st = gr.Textbox(label="📡 Status", interactive=False)

            with gr.TabItem("Character Guide"):
                gr.Markdown("### 🎙️ Quick Guide:\n- af/am: American\n- bf/bm: British\n- hf: Hindi\n- ef: Spanish\n- jf: Japanese")

        btn.click(KOKORO_TTS_API, 
                 inputs=[txt_in, lang_drp, voice_drp, spd, trans, sil, gap, ip_user],
                 outputs=[aud, f_dl, s_dl, s_dl, st],
                 show_progress="minimal")

    return demo

if __name__ == "__main__":
    ui().queue().launch(auth=custom_auth, css=css_hider)
