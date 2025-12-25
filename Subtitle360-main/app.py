# ====================================================================
# === APP.PY VERSION (SYNTAX ERROR FIXED) ===
# ====================================================================
import os
import subprocess
import sys

# Pip commands ko Python ke andar se chalane ka sahi tariqa
def install_requirements():
    try:
        import kokoro
        import gradio
    except ImportError:
        print("⏳ Installing requirements, please wait...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kokoro-onnx", "deep-translator", "gradio", "pydub", "kokoro>=0.1.9", "--quiet"])

# Requirements install karein
install_requirements()

from datetime import date
import json
import uuid
import re
import shutil
import wave
import numpy as np
import string
from pathlib import Path

# --- COLAB DIRECT SAVE & ASYNC FIX ---
from google.colab import files 
import nest_asyncio
nest_asyncio.apply()

from kokoro import KPipeline
import gradio as gr
from huggingface_hub import list_repo_files
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- 1. AUTHORIZED USERS ---
AUTHORIZED_USERS = {
    "Raza": {"password": "pass123", "expiry_date": date(2025, 12, 31)},      
    "ali_tts": {"password": "ali123", "expiry_date": date(2025, 12, 12)},
    "admin": {"password": "admin@123", "expiry_date": date(2099, 1, 1)},    
}
USER_IP_MAP_FILE = "user_ip_map.json"

def custom_auth(username, password):
    user_data = AUTHORIZED_USERS.get(username)
    if user_data and user_data.get("password") == password:
        user_expiry = user_data.get("expiry_date")
        if user_expiry and date.today() > user_expiry: return False 
        return True 
    return False

# --- 2. CORE FUNCTIONS ---
language_map = {"American English": "a", "British English": "b", "Hindi": "h"}

def update_pipeline(Language):
    global pipeline, last_used_language
    new_lang = language_map.get(Language, "a")
    if new_lang != last_used_language:
        pipeline = KPipeline(lang_code=new_lang)
        last_used_language = new_lang

def KOKORO_TTS_API(text, Language, voice, speed, ip_check_username, request: gr.Request):
    if not ip_check_username:
        return None, "Please enter Username"

    update_pipeline(Language)
    generator = pipeline(text, voice=voice, speed=speed)
    
    save_path = "output_audio.wav"
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for result in generator:
            audio_int16 = (result.audio.numpy() * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
    
    # --- DIRECT DOWNLOAD TRIGGER ---
    if os.path.exists(save_path):
        print(f"🚀 Triggering Direct Download...")
        files.download(save_path)
        
    return save_path, "✅ Generated & Saved to PC!"

# --- 3. UI & LAUNCH ---
with gr.Blocks(title="Long Touch TTS") as demo:
    gr.Markdown("# 🎙️ Kokoro TTS - Direct PC Save")
    with gr.Row():
        txt = gr.Textbox(label="Text", lines=3)
        usr = gr.Textbox(label="Username")
        lang = gr.Dropdown(list(language_map.keys()), value="American English")
        vc = gr.Dropdown(['af_heart', 'af_bella'], value="af_heart")
    btn = gr.Button("Generate", variant="primary")
    aud = gr.Audio()
    status = gr.Textbox(label="Status")

    btn.click(KOKORO_TTS_API, inputs=[txt, lang, vc, gr.Number(value=1, visible=False), usr], outputs=[aud, status])

if __name__ == "__main__":
    last_used_language = "a"
    pipeline = KPipeline(lang_code='a')
    demo.launch(share=True, auth=custom_auth)
