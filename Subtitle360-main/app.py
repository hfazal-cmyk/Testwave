# ====================================================================
# === FULLY FIXED CODE WITH DIRECT PC SAVE (NO SYNTAX ERROR) ===
# ====================================================================
from datetime import date, datetime
import json
import os
import uuid
import re
import shutil
import wave
import numpy as np
import string
import subprocess
from pathlib import Path

# --- COLAB FIXES ---
from google.colab import files 
import nest_asyncio
nest_asyncio.apply()

# Install Dependencies
!pip install kokoro-onnx deep-translator gradio pydub kokoro>=0.1.9 --quiet

from kokoro import KPipeline
import gradio as gr
from huggingface_hub import list_repo_files
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- 1. AUTHORIZED USERS & SECURITY ---
AUTHORIZED_USERS = {
    "Raza": {"password": "pass123", "expiry_date": date(2025, 12, 31)},      
    "ali_tts": {"password": "ali123", "expiry_date": date(2025, 12, 12)},
    "ali_pro": {"password": "ali789", "expiry_date": date(2026, 6, 15)},
    "admin": {"password": "admin@123", "expiry_date": date(2099, 1, 1)},    
}
USER_IP_MAP_FILE = "user_ip_map.json"

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
    if user_data and user_data.get("password") == password:
        user_expiry = user_data.get("expiry_date")
        if user_expiry and date.today() > user_expiry: return False 
        return True 
    return False

# --- 2. CORE TTS LOGIC ---
language_map = {
    "American English": "a", "British English": "b", "Hindi": "h", "Spanish": "e",
    "French": "f", "Italian": "i", "Brazilian Portuguese": "p", "Japanese": "j", "Mandarin Chinese": "z"
}

def update_pipeline(Language):
    global pipeline, last_used_language
    new_lang = language_map.get(Language, "a")
    if new_lang != last_used_language:
        pipeline = KPipeline(lang_code=new_lang)
        last_used_language = new_lang

def create_audio_dir():
    audio_dir = os.path.join(os.getcwd(), "kokoro_audio")
    if not os.path.exists(audio_dir): os.makedirs(audio_dir)
    return audio_dir

def clean_text(text):
    text = re.sub(r'[*#—–-]', ' ', text)
    emoji_pattern = re.compile(r'[\U0001F000-\U0001F9FF]|[\u2600-\u26FF]', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return re.sub(r'\s+', ' ', text).strip()

def generate_and_save_audio(text, Language, voice, speed, remove_silence, keep_silence_up_to):
    text = clean_text(text)
    update_pipeline(Language)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    
    temp_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
    save_path = os.path.join(temp_folder, temp_name)
    
    timestamps = {}
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for i, result in enumerate(generator):
            audio_np = result.audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
            timestamps[i] = {"duration": len(audio_np)/24000, "words": result.tokens}
            
    return save_path, timestamps

# --- 3. MAIN API FUNCTION (DIRECT SAVE TRIGGER) ---
def KOKORO_TTS_API(text, Language, voice, speed, translate_text, remove_silence, keep_silence_up_to, ip_check_username, request: gr.Request):
    if not ip_check_username:
        gr.Warning("Username Missing!")
        return None, None
    
    client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "UNKNOWN")
    ip_map = load_ip_map()
    username = ip_check_username.strip().lower()
    
    if username in ip_map and ip_map[username] != client_ip:
        gr.Warning("IP Locked!")
        return None, None
    elif username not in ip_map:
        ip_map[username] = client_ip
        save_ip_map(ip_map)

    if translate_text:
        text = GoogleTranslator(target='en').translate(text)

    save_path, _ = generate_and_save_audio(text, Language, voice, speed, remove_silence, keep_silence_up_to)
    
    # --- DIRECT DOWNLOAD ---
    if save_path and os.path.exists(save_path):
        print(f"🚀 Downloading to PC: {save_path}")
        files.download(save_path)
        
    return save_path, save_path

# --- 4. GRADIO UI ---
def ui():
    with gr.Blocks(title="Long Touch TTS", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# 🎙️ Kokoro TTS - Direct Save Mode")
        
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(label="Enter Text", lines=4)
                usr = gr.Textbox(label="🔒 Security Username")
                lang = gr.Dropdown(list(language_map.keys()), value="American English", label="Language")
                vc = gr.Dropdown(['af_heart', 'af_bella', 'af_nicole', 'af_sky', 'am_adam'], value="af_heart", label="Voice")
                btn = gr.Button("Generate & Save to PC", variant="primary")
            
            with gr.Column():
                aud = gr.Audio(label="Preview")
                fl = gr.File(label="Backup Download")

        btn.click(KOKORO_TTS_API, inputs=[txt, lang, vc, gr.Number(value=1, visible=False), 
                                         gr.Checkbox(value=False, visible=False), 
                                         gr.Checkbox(value=False, visible=False), 
                                         gr.Number(value=0.05, visible=False), usr], outputs=[aud, fl])
    return demo

# --- 5. INITIALIZE & LAUNCH ---
last_used_language = "a"
pipeline = KPipeline(lang_code='a')
temp_folder = create_audio_dir()

if __name__ == "__main__":
    gr.close_all()
    app = ui()
    app.launch(share=True, auth=custom_auth, debug=True)
