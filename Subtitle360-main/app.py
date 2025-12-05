# ====================================================================
# === 1. MULTI-USER DICTIONARY, EXPIRY DATE, & IP MAP FILE ADDED ===
# ====================================================================
from datetime import date 
import json 


AUTHORIZED_USERS = {
    "user1": {
        "password": "pass123", 
        "expiry_date": date(2025, 12, 31) 
    },      
    "ali_tts": {
        "password": "ali123", 
        "expiry_date": date(2024, 12, 1) 
    },
    "ali_pro": {
        "password": "ali789", 
        "expiry_date": date(2026, 6, 15) 
    },
    "guest_04": {
        "password": "gpass",
        "expiry_date": date(2025, 1, 1) 
    },
    "admin": {
        "password": "admin@123",
        "expiry_date": date(2099, 1, 1) 
    },    
    
    "client50": {
        "password": "secretpass",
        "expiry_date": date(2025, 3, 31)
    }, 
}


USER_IP_MAP_FILE = "user_ip_map.json"
# ====================================================================

# Initalize a pipeline
from kokoro import KPipeline
# from IPython.display import display, Audio
# import soundfile as sf
import os
from huggingface_hub import list_repo_files
import uuid
import re 
import gradio as gr
import shutil

# ====================================================================
# === IP MAP HELPER FUNCTIONS (Wohi hain, koi badlaav nahi) ===
# ====================================================================
def load_ip_map():
    """Loads the username-IP mapping from the JSON file."""
    if os.path.exists(USER_IP_MAP_FILE):
        try:
            with open(USER_IP_MAP_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Agar file empty ya corrupt ho, toh nayi dictionary return karein
            return {}
    return {}

def save_ip_map(ip_map):
    """Saves the username-IP mapping to the JSON file."""
    try:
        with open(USER_IP_MAP_FILE, 'w') as f:
            json.dump(ip_map, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving IP map: {e}")
        return False

# ====================================================================
# === 2. custom_auth Function Updated (Individual Expiry Check) ===
# ====================================================================
def custom_auth(username, password):
    """
    Checks if the provided username and password match any authorized user
    and verifies the INDIVIDUAL user's expiry date.
    """
    user_data = AUTHORIZED_USERS.get(username)
    
    # 1. User Existence Check
    if user_data is None:
        print(f"Login failed for {username}: User not found.")
        return False

    # 2. Password Check
    if user_data.get("password") == password:
        
        # 3. INDIVIDUAL Expiry Date Check
        user_expiry = user_data.get("expiry_date")
        
        if user_expiry and date.today() > user_expiry:
            # Access expired
            print(f"Login failed for {username}: Access expired on {user_expiry.strftime('%Y-%m-%d')}.")
            # User ko batane ke liye ki uska access kab tak tha
            return False 
        
        # All checks passed
        print(f"Login successful for user: {username}")
        return True 
        
    # Credentials mismatch
    print(f"Login failed for {username}: Invalid credentials.")
    return False 
# ====================================================================


# === HIDING GITHUB FOOTER & LINKS (Wohi hai) ===
css_hider = """
/* Gradio ka footer aur 'Made with Gradio' chhipane ke liye */
footer { visibility: hidden !important; height: 0px !important; }
/* Container ki height adjust karne ke liye */
.gradio-container { min-height: 0px !important; }
"""
# =============================================================

#translate langauge 
from deep_translator import GoogleTranslator
def bulk_translate(text, target_language, chunk_size=500):
    language_map_local = {
    "American English": "en",  
    "British English": "en",  
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Brazilian Portuguese": "pt",
    "Japanese": "ja",
    "Mandarin Chinese": "zh-CN"
    }
    # lang_code = GoogleTranslator().get_supported_languages(as_dict=True).get(target_language.lower())
    lang_code=language_map_local[target_language]
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    translated_chunks = [GoogleTranslator(target=lang_code).translate(chunk) for chunk in chunks]
    result=" ".join(translated_chunks)
    return result.strip()
    
# Language mapping dictionary
language_map = {
    "American English": "a",
    "British English": "b",
    "Hindi": "h",
    "Spanish": "e",
    "French": "f",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "Japanese": "j",
    "Mandarin Chinese": "z"
}


def update_pipeline(Language):
    """ Updates the pipeline only if the language has changed. """
    global pipeline, last_used_language
    # Get language code, default to 'a' if not found
    new_lang = language_map.get(Language, "a")

    # Only update if the language is different
    if new_lang != last_used_language:
        pipeline = KPipeline(lang_code=new_lang)
        last_used_language = new_lang 
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang  # Update last used language
        except Exception as e:
            gr.Warning(f"Make sure the input text is in {Language}",duration=10)
            gr.Warning(f"Fallback to English Language",duration=5)
            pipeline = KPipeline(lang_code="a")  # Fallback to English
            last_used_language = "a"



def get_voice_names(repo_id):
    """Fetches and returns a list of voice names (without extensions) from the given Hugging Face repository."""
    return [os.path.splitext(file.replace("voices/", ""))[0] for file in list_repo_files(repo_id) if file.startswith("voices/")]

def create_audio_dir():
    """Creates the 'kokoro_audio' directory in the root folder if it doesn't exist."""
    root_dir = os.getcwd()  # Use current working directory instead of __file__
    audio_dir = os.path.join(root_dir, "kokoro_audio")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    else:
        print(f"Directory already exists: {audio_dir}")
    return audio_dir

import re

def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "-": " ",  # Replace hyphen with space
        "**": " ", # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous symbols and pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and map symbols
        r'[\U0001F700-\U0001F77F]|'  # Alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric shapes extended
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental symbols and pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and pictographs extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U0001F1E0-\U0001F1FF]'  # Flags (iOS)
        r'', flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tts_file_name(text,language):
    global temp_folder
    # Remove all non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Retain only alphabets and spaces
    text = text.lower().strip()          # Convert to lowercase and strip leading/trailing spaces
    text = text.replace(" ", "_")        # Replace spaces with underscores
    language=language.replace(" ", "_").strip() 
    # Truncate or handle empty text
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else language
    
    # Generate a random string for uniqueness
    random_string = uuid.uuid4().hex[:8].upper()
    
    # Construct the file name
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name


# import soundfile as sf
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                     min_silence_len=100,
                                     silence_thresh=-45,
                                     keep_silence=minimum_silence)  
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path

def generate_and_save_audio(text, Language="American English",voice="af_bella", speed=1,remove_silence=False,keep_silence_up_to=0.05):
    text=clean_text(text)
    update_pipeline(Language)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    save_path=tts_file_name(text,Language)
    # Open the WAV file for writing
    timestamps={}
    with wave.open(save_path, 'wb') as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(24000)  # Sample rate
        for i, result in enumerate(generator):
            gs = result.graphemes # str
        #    print(f"\n{i}: {gs}")
            ps = result.phonemes # str
            # audio = result.audio.cpu().numpy()
            audio = result.audio
            tokens = result.tokens # List[en.MToken]
            timestamps[i]={"text":gs,"words":[]}
            if Language in ["American English", "British English"]:
                for t in tokens:
                    # print(t.text, repr(t.whitespace), t.start_ts, t.end_ts)
                    timestamps[i]["words"].append({"word":t.text,"start":t.start_ts,"end":t.end_ts})
            audio_np = audio.numpy()  # Convert Tensor to NumPy array
            audio_int16 = (audio_np * 32767).astype(np.int16)  # Scale to 16-bit range
            audio_bytes = audio_int16.tobytes()  # Convert to bytes
            # Write the audio chunk to the WAV file
            duration_sec = len(audio_np) / 24000
            timestamps[i]["duration"] = duration_sec
            wav_file.writeframes(audio_bytes)
    if remove_silence:        
        keep_silence = int(keep_silence_up_to * 1000)
        new_wave_file=remove_silence_function(save_path,minimum_silence=keep_silence)
        return new_wave_file,timestamps
    return save_path,timestamps



def adjust_timestamps(timestamp_dict):
    adjusted_timestamps = []
    last_global_end = 0  # Cumulative audio timeline

    for segment_id in sorted(timestamp_dict.keys()):
        segment = timestamp_dict[segment_id]
        words = segment["words"]
        chunk_duration = segment["duration"]

        # If there are valid words, get last word end
        last_word_end_in_chunk = (
            max(w["end"] for w in words if w["end"] not in [None, 0])
            if words else 0
        )

        silence_gap = chunk_duration - last_word_end_in_chunk
        if silence_gap < 0:  # In rare cases where end > duration (due to rounding)
            silence_gap = 0

        for word in words:
            start = word["start"] or 0
            end = word["end"] or start

            adjusted_timestamps.append({
                "word": word["word"],
                "start": round(last_global_end + start, 3),
                "end": round(last_global_end + end, 3)
            })

        # Add entire chunk duration to global end
        last_global_end += chunk_duration

    return adjusted_timestamps



import string

def write_word_srt(word_level_timestamps, output_file="word.srt", skip_punctuation=True):
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1  # Track subtitle numbering separately

        for entry in word_level_timestamps:
            word = entry["word"]
            
            # Skip punctuation if enabled
            if skip_punctuation and all(char in string.punctuation for char in word):
                continue

            start_time = entry["start"]
            end_time = entry["end"]

            # Convert seconds to SRT time format (HH:MM:SS,mmm)
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                sec = int(seconds % 60)
