# ====================================================================
# === 1. UPDATED WITH DIRECT PC SAVE (Colab High Speed) ===
# ====================================================================
from datetime import date 
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

# --- COLAB DIRECT DOWNLOAD FIX ---
from google.colab import files  # Ye line PC mein direct save karegi
import nest_asyncio
nest_asyncio.apply()

from kokoro import KPipeline
import gradio as gr
from huggingface_hub import list_repo_files
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence

# AUTHORIZED_USERS (Wohi purana data)
AUTHORIZED_USERS = {
    "Raza": {"password": "pass123", "expiry_date": date(2025, 12, 31)},      
    "ali_tts": {"password": "ali123", "expiry_date": date(2025, 12, 12)},
    "ali_pro": {"password": "ali789", "expiry_date": date(2026, 6, 15)},
    "admin": {"password": "admin@123", "expiry_date": date(2099, 1, 1)},    
}

USER_IP_MAP_FILE = "user_ip_map.json"

# [Baqi Helper Functions: load_ip_map, save_ip_map, custom_auth, bulk_translate etc. same hain]
def load_ip_map():
    if os.path.exists(USER_IP_MAP_FILE):
        try:
            with open(USER_IP_MAP_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_ip_map(ip_map):
    try:
        with open(USER_IP_MAP_FILE, 'w') as f:
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

# Language mapping
language_map = {"American English": "a", "British English": "b", "Hindi": "h", "Spanish": "e", "French": "f", "Italian": "i", "Brazilian Portuguese": "p", "Japanese": "j", "Mandarin Chinese": "z"}

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

# --- KOKORO_TTS_API (MAIN UPDATE YAHAN HAI) ---
def KOKORO_TTS_API(text, Language="American English", voice="af_bella", speed=1, translate_text=False, remove_silence=False, keep_silence_up_to=0.05, ip_check_username: str = None, request: gr.Request = None):
    if not ip_check_username or request is None:
        gr.Warning("Access Denied: Enter Username", duration=7)
        return None, None, None, None, None

    client_ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "UNKNOWN")
    ip_map = load_ip_map()
    username = ip_check_username.strip().lower()
    
    if username in ip_map:
        if client_ip != ip_map[username]:
            gr.Warning(f"Locked to IP: {ip_map[username]}", duration=10)
            return None, None, None, None, None
    else:
        ip_map[username] = client_ip
        save_ip_map(ip_map)

    if translate_text: text = bulk_translate(text, Language)
    
    # Audio Banana
    save_path, timestamps = generate_and_save_audio(text=text, Language=Language, voice=voice, speed=speed, remove_silence=remove_silence, keep_silence_up_to=keep_silence_up_to)
    
    word_level_srt, normal_srt, json_file = None, None, None

    if remove_silence == False and Language in ["American English", "British English"]:
        word_level_timestamps = adjust_timestamps(timestamps)
        word_level_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="word_level_")
        normal_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="sentence_")
        json_file = modify_filename(save_path.replace(".wav", ".json"), prefix="duration_")
        
        write_word_srt(word_level_timestamps, output_file=word_level_srt)
        write_sentence_srt(word_level_timestamps, output_file=normal_srt)
        make_json(word_level_timestamps, json_file)
        
        save_current_data()
        for f in [save_path, word_level_srt, normal_srt, json_file]:
            if f: shutil.copy(f, "./last/")

    # === DIRECT PC SAVE TRIGGER ===
    if save_path and os.path.exists(save_path):
        print(f"🚀 Audio Ready! Saving to PC at Full Speed...")
        files.download(save_path) # Direct Browser Download
        if normal_srt and os.path.exists(normal_srt):
            files.download(normal_srt) # Auto-Download SRT also

    return save_path, save_path, word_level_srt, normal_srt, json_file

# [Baqi function: clean_text, tts_file_name, remove_silence_function, generate_and_save_audio etc. same rahenge jo pehle thay]
# (Yahan logic continue hoga...)
