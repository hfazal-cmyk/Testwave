import os
import re
import sys
import json
import time
import uuid
import wave
import shutil
import string
import numpy as np
import subprocess
from datetime import date
from pathlib import Path

# --- COLAB DIRECT SAVE FIX ---
from google.colab import files 
import nest_asyncio
nest_asyncio.apply()

# 1. INSTALL NECESSARY LIBS
!pip install kokoro-onnx deep-translator gradio pydub --quiet

from kokoro import KPipeline
import gradio as gr
from huggingface_hub import list_repo_files
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- AUTH & SECURITY (Wohi purana) ---
AUTHORIZED_USERS = {
    "Raza": {"password": "pass123", "expiry_date": date(2025, 12, 31)},      
    "ali_tts": {"password": "ali123", "expiry_date": date(2025, 12, 12)},
    "ali_pro": {"password": "ali789", "expiry_date": date(2026, 6, 15)},
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

# ... [Baqi sare Helper Functions jaise load_ip_map, save_ip_map yahan backend mein chalenge] ...

# --- MAIN GENERATION FUNCTION WITH AUTO-SAVE ---
def KOKORO_TTS_API(text, Language, voice, speed, translate_text, remove_silence, keep_silence_up_to, ip_check_username, request: gr.Request):
    # IP Check Logic
    client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "UNKNOWN")
    # (IP validation yahan same wahi hogi jo aapne pehle di thi)

    if translate_text:
        text = GoogleTranslator(target='en').translate(text) # Example simplification

    # Audio Banane ka process
    save_path, timestamps = generate_and_save_audio(text, Language, voice, speed, remove_silence, keep_silence_up_to)
    
    # --- ASAL KAMAAL YAHAN HAI ---
    if save_path and os.path.exists(save_path):
        print(f"🚀 Direct Downloading: {save_path}")
        files.download(save_path) # Ye line PC mein save karegi bina kisi link ke!
        
    return save_path, save_path, None, None, None

# --- RE-WRITTEN LAUNCHER (Link Issue Fix) ---
def launch_app():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("## 🎙️ Kokoro TTS - Direct PC Save Edition")
        
        with gr.Row():
            with gr.Column():
                txt_input = gr.Textbox(label="Enter Text", lines=3)
                user_input = gr.Textbox(label="🔒 Username for IP Lock")
                lang_drop = gr.Dropdown(['American English', 'British English', 'Hindi'], value='American English', label="Language")
                voice_drop = gr.Dropdown(['af_heart', 'af_bella'], value='af_heart', label="Voice")
                gen_btn = gr.Button("Generate & Save to PC", variant="primary")
            
            with gr.Column():
                audio_out = gr.Audio(label="Preview", autoplay=True)
                file_out = gr.File(label="Manual Download Link")

        gen_btn.click(
            KOKORO_TTS_API, 
            inputs=[txt_input, lang_drop, voice_drop, gr.Number(value=1, visible=False), 
                    gr.Checkbox(value=False, visible=False), gr.Checkbox(value=False, visible=False), 
                    gr.Number(value=0.05, visible=False), user_input],
            outputs=[audio_out, file_out]
        )

    # PORT aur SERVER cleanup taake link generate ho
    gr.close_all()
    print("⏳ Wait 10 seconds for the Public Link...")
    demo.queue().launch(share=True, auth=custom_auth, debug=True, show_api=False)

# Start logic
if __name__ == "__main__":
    create_audio_dir()
    launch_app()
