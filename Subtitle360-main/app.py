# ====================================================================
# === 1. MULTI-USER DICTIONARY, EXPIRY DATE, & IP MAP FILE ADDED ===
# ====================================================================
from datetime import date 
import json 


AUTHORIZED_USERS = {
    "Raza": {
        "password": "pass123", 
        "expiry_date": date(2025, 12, 31) 
    },      
    "ali_tts": {
        "password": "ali123", 
        "expiry_date": date(2025, 12, 12) 
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
import os
from huggingface_hub import list_repo_files
import uuid
import re 
import gradio as gr
import shutil

# ====================================================================
# === IP MAP HELPER FUNCTIONS ===
# ====================================================================
def load_ip_map():
    """Loads the username-IP mapping from the JSON file."""
    if os.path.exists(USER_IP_MAP_FILE):
        try:
            with open(USER_IP_MAP_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
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
# === 2. custom_auth Function Updated ===
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
            print(f"Login failed for {username}: Access expired on {user_expiry.strftime('%Y-%m-%d')}.")
            return False 
        
        # All checks passed
        print(f"Login successful for user: {username}")
        return True 
        
    # Credentials mismatch
    print(f"Login failed for {username}: Invalid credentials.")
    return False 
# ====================================================================

# === CSS HIDER ===
css_hider = """
/* Gradio ka footer aur 'Made with Gradio' chhipane ke liye */
footer { visibility: hidden !important; height: 0px !important; }
/* Container ki height adjust karne ke liye */
.gradio-container { min-height: 0px !important; }
/* Small file box for better UI */
.small-file-box { 
    height: 75px !important; 
    min-height: 75px !important; 
    overflow: hidden !important; 
    border: 1px solid #ddd !important; 
    margin-top: 5px !important;
}
"""
# =============================================================

# ====================================================================
# === NEW: VOICE GROUPING AND TRANSLATION CODES ===
# ====================================================================
VOICE_GROUPS = {
    "af": "American Female", "am": "American Male",
    "bf": "British Female", "bm": "British Male",
    "hf": "Hindi Female", "hm": "Hindi Male",
    "ef": "Spanish Female", "em": "Spanish Male",
    "ff": "French Female", "if": "Italian Female",
    "pf": "Portuguese Female", "jf": "Japanese Female",
    "zf": "Chinese Female"
}

TRANS_CODES = {
    "American English": "en", "British English": "en", "Hindi": "hi",
    "Spanish": "es", "French": "fr", "Italian": "it",
    "Brazilian Portuguese": "pt", "Japanese": "ja", "Mandarin Chinese": "zh-CN"
}
# ====================================================================

# ====================================================================
# === TRANSLATION FUNCTION ===
# ====================================================================
from deep_translator import GoogleTranslator

def bulk_translate(text, target_language, chunk_size=500):
    """Translate text to target language using Google Translator."""
    lang_code = TRANS_CODES.get(target_language, "en")
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
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

    translated_chunks = []
    for chunk in chunks:
        try:
            translated = GoogleTranslator(target=lang_code).translate(chunk)
            translated_chunks.append(translated)
        except:
            translated_chunks.append(chunk)
    
    result = " ".join(translated_chunks)
    return result.strip()

# ====================================================================
# === LANGUAGE MAPPING ===
# ====================================================================
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

# ====================================================================
# === PIPELINE MANAGEMENT ===
# ====================================================================
pipeline = None
last_used_language = "a"
temp_folder = ""

def update_pipeline(Language):
    """Updates the pipeline only if the language has changed."""
    global pipeline, last_used_language
    new_lang = language_map.get(Language, "a")
    
    if new_lang != last_used_language:
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang
        except Exception as e:
            print(f"Error updating pipeline: {e}")
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"

# ====================================================================
# === VOICE FUNCTIONS ===
# ====================================================================
def get_readable_voices():
    """Returns voices with human-readable names."""
    raw = [os.path.splitext(f.replace("voices/", ""))[0] 
           for f in list_repo_files("hexgrad/Kokoro-82M") 
           if f.startswith("voices/")]
    
    choices = []
    for v in raw:
        prefix = v[:2]
        name = v[3:].capitalize() if len(v) > 2 else v
        desc = VOICE_GROUPS.get(prefix, prefix.upper())
        choices.append((f"{desc} {name}", v))
    
    return sorted(choices)

def get_voice_names(repo_id):
    """Fetches and returns a list of voice names."""
    return [os.path.splitext(file.replace("voices/", ""))[0] 
            for file in list_repo_files(repo_id) 
            if file.startswith("voices/")]

# ====================================================================
# === AUDIO DIRECTORY ===
# ====================================================================
def create_audio_dir():
    """Creates the 'kokoro_audio' directory."""
    root_dir = os.getcwd()
    audio_dir = os.path.join(root_dir, "kokoro_audio")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    else:
        print(f"Directory already exists: {audio_dir}")
    return audio_dir

# ====================================================================
# === TEXT CLEANING ===
# ====================================================================
def clean_text(text):
    """Cleans text by removing unwanted characters and emojis."""
    replacements = {
        "–": " ", "-": " ", "**": " ", "*": " ", "#": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'
        r'[\U0001F300-\U0001F5FF]|'
        r'[\U0001F680-\U0001F6FF]|'
        r'[\U0001F700-\U0001F77F]|'
        r'[\U0001F780-\U0001F7FF]|'
        r'[\U0001F800-\U0001F8FF]|'
        r'[\U0001F900-\U0001F9FF]|'
        r'[\U0001FA00-\U0001FA6F]|'
        r'[\U0001FA70-\U0001FAFF]|'
        r'[\U00002702-\U000027B0]|'
        r'[\U0001F1E0-\U0001F1FF]',
        flags=re.UNICODE
    )
    
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ====================================================================
# === FILE NAME GENERATION ===
# ====================================================================
def tts_file_name(text, language):
    """Generates a unique filename for TTS output."""
    global temp_folder
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = text.replace(" ", "_")
    language = language.replace(" ", "_").strip()
    
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else language
    random_string = uuid.uuid4().hex[:8].upper()
    
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name

# ====================================================================
# === AUDIO PROCESSING ===
# ====================================================================
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence_function(file_path, minimum_silence=50):
    """Removes silence from audio file."""
    output_path = file_path.replace(".wav", "_no_silence.wav")
    sound = AudioSegment.from_file(file_path, format="wav")
    audio_chunks = split_on_silence(
        sound,
        min_silence_len=100,
        silence_thresh=-45,
        keep_silence=minimum_silence
    )
    
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    
    combined.export(output_path, format="wav")
    return output_path

def generate_and_save_audio(text, Language="American English", voice="af_bella", speed=1, remove_silence=False, keep_silence_up_to=0.05):
    """Generates and saves TTS audio."""
    text = clean_text(text)
    update_pipeline(Language)
    
    if pipeline is None:
        pipeline = KPipeline(lang_code=language_map.get(Language, "a"))
    
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    save_path = tts_file_name(text, Language)
    
    timestamps = {}
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        
        for i, result in enumerate(generator):
            gs = result.graphemes
            audio = result.audio
            tokens = result.tokens
            
            timestamps[i] = {"text": gs, "words": []}
            
            if Language in ["American English", "British English"]:
                for t in tokens:
                    timestamps[i]["words"].append({
                        "word": t.text,
                        "start": t.start_ts,
                        "end": t.end_ts
                    })
            
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            duration_sec = len(audio_np) / 24000
            timestamps[i]["duration"] = duration_sec
            wav_file.writeframes(audio_bytes)
    
    if remove_silence:
        keep_silence = int(keep_silence_up_to * 1000)
        new_wave_file = remove_silence_function(save_path, minimum_silence=keep_silence)
        return new_wave_file, timestamps
    
    return save_path, timestamps

# ====================================================================
# === TIMESTAMP ADJUSTMENT ===
# ====================================================================
def adjust_timestamps(timestamp_dict):
    """Adjusts timestamps for word-level alignment."""
    adjusted_timestamps = []
    last_global_end = 0

    for segment_id in sorted(timestamp_dict.keys()):
        segment = timestamp_dict[segment_id]
        words = segment["words"]
        chunk_duration = segment["duration"]

        last_word_end_in_chunk = (
            max(w["end"] for w in words if w["end"] not in [None, 0])
            if words else 0
        )

        silence_gap = chunk_duration - last_word_end_in_chunk
        if silence_gap < 0:
            silence_gap = 0

        for word in words:
            start = word["start"] or 0
            end = word["end"] or start

            adjusted_timestamps.append({
                "word": word["word"],
                "start": round(last_global_end + start, 3),
                "end": round(last_global_end + end, 3)
            })

        last_global_end += chunk_duration

    return adjusted_timestamps

# ====================================================================
# === SUBTITLE FUNCTIONS ===
# ====================================================================
import string

def write_word_srt(word_level_timestamps, output_file="word.srt", skip_punctuation=True):
    """Writes word-level SRT file."""
    with open(output_file, "w", encoding="utf-8") as f:
        index = 1

        for entry in word_level_timestamps:
            word = entry["word"]
            
            if skip_punctuation and all(char in string.punctuation for char in word):
                continue

            start_time = entry["start"]
            end_time = entry["end"]

            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                sec = int(seconds % 60)
                millisec = int((seconds % 1) * 1000)
                return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"

            start_srt = format_srt_time(start_time)
            end_srt = format_srt_time(end_time)

            f.write(f"{index}\n{start_srt} --> {end_srt}\n{word}\n\n")
            index += 1

def split_line_by_char_limit(text, max_chars=30):
    """Splits text into lines with character limit."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_chars:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        if len(current_line.split()) == 1 and len(lines) > 0:
            lines[-1] += " " + current_line
        else:
            lines.append(current_line)

    return "\n".join(lines)

def write_sentence_srt(word_level_timestamps, output_file="subtitles.srt", max_words=8, min_pause=0.1):
    """Writes sentence-level SRT file."""
    subtitles = []
    subtitle_words = []
    start_time = None
    remove_punctuation = ['"', "—"]

    for i, entry in enumerate(word_level_timestamps):
        word = entry["word"]
        word_start = entry["start"]
        word_end = entry["end"]

        if word in remove_punctuation:
            continue

        if word in string.punctuation:
            if subtitle_words:
                subtitle_words[-1] = (subtitle_words[-1][0] + word, subtitle_words[-1][1])
            continue

        if start_time is None:
            start_time = word_start

        if subtitle_words:
            last_word_end = subtitle_words[-1][1]
            pause_duration = word_start - last_word_end
        else:
            pause_duration = 0

        if (word.endswith(('.', '!', '?')) and len(subtitle_words) >= 5) or len(subtitle_words) >= max_words or pause_duration > min_pause:
            end_time = subtitle_words[-1][1]
            subtitle_text = " ".join(w[0] for w in subtitle_words)
            subtitles.append((start_time, end_time, subtitle_text))

            subtitle_words = [(word, word_end)]
            start_time = word_start
            continue

        subtitle_words.append((word, word_end))

    if subtitle_words:
        end_time = subtitle_words[-1][1]
        subtitle_text = " ".join(w[0] for w in subtitle_words)
        subtitles.append((start_time, end_time, subtitle_text))

    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        millisec = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, start=1):
            text = split_line_by_char_limit(text, max_chars=30)
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

# ====================================================================
# === JSON FUNCTIONS ===
# ====================================================================
def fix_punctuation(text):
    """Fixes punctuation spacing in text."""
    text = re.sub(r'\s([.,?!])', r'\1', text)
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = text.replace('" ', '"')
    
    track = 0
    result = []
    
    for index, char in enumerate(text):
        if char == '"':
            track += 1
            result.append(char)
            if track % 2 == 0:
                result.append(' ')
        else:
            result.append(char)
    
    text = ''.join(result)
    return text.strip()

def make_json(word_timestamps, json_file_name):
    """Creates JSON file with sentence timestamps."""
    data = {}
    temp = []
    inside_quote = False
    start_time = word_timestamps[0]['start']
    end_time = word_timestamps[0]['end']
    words_in_sentence = []
    sentence_id = 0

    for i, word_data in enumerate(word_timestamps):
        word = word_data['word']
        word_start = word_data['start']
        word_end = word_data['end']

        words_in_sentence.append({'word': word, 'start': word_start, 'end': word_end})
        end_time = word_end

        if word == '"':
            if inside_quote:
                temp[-1] += '"'
            else:
                temp.append('"')
            inside_quote = not inside_quote
        else:
            temp.append(word)

        if word.endswith(('.', '?', '!')) and not inside_quote:
            if i + 1 < len(word_timestamps):
                next_word = word_timestamps[i + 1]['word']
                if next_word[0].islower():
                    continue

            sentence = " ".join(temp)
            sentence = fix_punctuation(sentence)
            data[sentence_id] = {
                'text': sentence,
                'duration': end_time - start_time,
                'start': start_time,
                'end': end_time,
                'words': words_in_sentence
            }

            temp = []
            words_in_sentence = []
            start_time = word_data['start']
            sentence_id += 1

    if temp:
        sentence = " ".join(temp)
        sentence = fix_punctuation(sentence)
        data[sentence_id] = {
            'text': sentence,
            'duration': end_time - start_time,
            'start': start_time,
            'end': end_time,
            'words': words_in_sentence
        }

    with open(json_file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    return json_file_name

# ====================================================================
# === FILE UTILITIES ===
# ====================================================================
def modify_filename(save_path: str, prefix: str = ""):
    """Modifies filename with prefix."""
    directory, filename = os.path.split(save_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{prefix}{name}{ext}"
    return os.path.join(directory, new_filename)

def save_current_data():
    """Saves current data to last folder."""
    if os.path.exists("./last"):
        shutil.rmtree("./last")
    os.makedirs("./last", exist_ok=True)

# ====================================================================
# === MAIN TTS API FUNCTION ===
# ====================================================================
def KOKORO_TTS_API(text, Language="American English", voice="af_bella", speed=1, translate_text=False, 
                   remove_silence=False, keep_silence_up_to=0.05, ip_check_username: str = None, 
                   request: gr.Request = None, progress=gr.Progress()):
    """
    Main TTS function with progress bar and all features.
    """
    # === IP/Security Check ===
    if not ip_check_username:
        gr.Warning("Access Denied: Please enter your Username.", duration=7)
        return None, None, None, None, None, "⚠️ Username Enter Karein!"

    if request is None:
        gr.Warning("Error: Could not retrieve client information.", duration=5)
        return None, None, None, None, None, "⚠️ Client info missing"

    client_ip = request.headers.get("x-forwarded-for", "UNKNOWN")
    if client_ip == "UNKNOWN":
        client_ip = request.client.host if request.client else "UNKNOWN"
    
    if client_ip == "UNKNOWN":
        gr.Warning("Error: IP address could not be determined.", duration=5)
        return None, None, None, None, None, "⚠️ IP address missing"

    ip_map = load_ip_map()
    username = ip_check_username.strip().lower()
    
    if username in ip_map:
        allowed_ip = ip_map[username]
        if client_ip != allowed_ip:
            error_msg = f"Access Denied: IP mismatch. Locked to {allowed_ip}"
            gr.Warning(error_msg, duration=10)
            return None, None, None, None, None, f"⚠️ {error_msg}"
        else:
            gr.Info(f"IP check successful for {username}.", duration=3)
    else:
        ip_map[username] = client_ip
        if save_ip_map(ip_map):
            gr.Info(f"First generation for {username}. IP locked.", duration=7)
        else:
            gr.Warning("Warning: Could not save IP lock.", duration=5)

    # === Progress Updates ===
    progress(0.1, desc="Preparing text...")
    
    if translate_text:
        progress(0.2, desc=f"Translating to {Language}...")
        text = bulk_translate(text, Language, chunk_size=500)
    
    progress(0.3, desc=f"Generating {Language} audio...")
    save_path, timestamps = generate_and_save_audio(
        text=text, 
        Language=Language, 
        voice=voice, 
        speed=speed, 
        remove_silence=remove_silence, 
        keep_silence_up_to=keep_silence_up_to
    )
    
    # === Create Output Files ===
    if not remove_silence and Language in ["American English", "British English"]:
        progress(0.6, desc="Creating subtitles...")
        word_level_timestamps = adjust_timestamps(timestamps)
        
        word_level_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="word_level_")
        normal_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="sentence_")
        json_file = modify_filename(save_path.replace(".wav", ".json"), prefix="duration_")
        
        write_word_srt(word_level_timestamps, output_file=word_level_srt, skip_punctuation=True)
        write_sentence_srt(word_level_timestamps, output_file=normal_srt, min_pause=0.01)
        make_json(word_level_timestamps, json_file)
        
        progress(0.8, desc="Saving files...")
        save_current_data()
        shutil.copy(save_path, "./last/")
        shutil.copy(word_level_srt, "./last/")
        shutil.copy(normal_srt, "./last/")
        shutil.copy(json_file, "./last/")
        
        progress(1.0, desc="Complete!")
        return save_path, save_path, word_level_srt, normal_srt, json_file, f"✅ Success! User: {username}"
    
    progress(1.0, desc="Complete!")
    return save_path, save_path, None, None, None, f"✅ Success! User: {username}"

# ====================================================================
# === UI FUNCTIONS ===
# ====================================================================
def load_text_from_file(file):
    """Load text from uploaded file."""
    if file is not None:
        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return "Error reading file"
    return ""

def toggle_autoplay(autoplay):
    """Toggle autoplay for audio."""
    return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

def ui():
    """Main UI function."""
    voice_choices = get_readable_voices()
    
    with gr.Blocks(title="Long Touch Generator 03060914996", css=css_hider) as demo:
        gr.Markdown("# 🎙️ Long Touch Generator 03060914996")
        
        with gr.Tabs():
            with gr.TabItem("Text To Speech"):
                with gr.Row():
                    with gr.Column():
                        # File upload section
                        with gr.Accordion("📄 Upload Script File (.txt)", open=False):
                            file_input = gr.File(label="Select File", file_types=[".txt"], type="filepath")
                        
                        # Main text input
                        text_input = gr.Textbox(
                            label='📝 Enter Text', 
                            lines=8, 
                            placeholder='Write or paste your script here...',
                            info='You can type directly or upload a .txt file above'
                        )
                        
                        # Update text when file is uploaded
                        file_input.change(
                            fn=load_text_from_file,
                            inputs=[file_input],
                            outputs=[text_input]
                        )
                        
                        # Security input
                        ip_check_username_input = gr.Textbox(
                            label='🔒 User Name Enter', 
                            placeholder='Enter username (e.g., user1, ali_tts)',
                            info='Your account will be locked to this IP address'
                        )

                        # Language and voice selection
                        with gr.Row():
                            language_name = gr.Dropdown(
                                list(language_map.keys()), 
                                label="🌍 Select Language", 
                                value="American English",
                                info="9 supported languages"
                            )
                            
                            voice_name = gr.Dropdown(
                                choices=voice_choices,
                                label="🎙️ Choose Character", 
                                value="af_heart",
                                info="50+ voices available"
                            )

                        # Generate button
                        generate_btn = gr.Button('🚀 Generate Audio', variant='primary', size="lg")

                        # Audio settings
                        with gr.Accordion('🎛️ Audio Settings', open=False):
                            speed = gr.Slider(
                                minimum=0.25, 
                                maximum=2, 
                                value=1, 
                                step=0.1, 
                                label='⚡️ Speed'
                            )
                            translate_text = gr.Checkbox(
                                value=False, 
                                label='🌐 Auto-Translate'
                            )
                            remove_silence = gr.Checkbox(
                                value=False, 
                                label='✂️ Remove Silence'
                            )
                            keep_silence_up_to = gr.Slider(
                                minimum=0.01, 
                                maximum=0.5, 
                                value=0.05, 
                                step=0.01, 
                                label='Quiet Gap Size (Seconds)'
                            )
                            autoplay_checkbox = gr.Checkbox(
                                value=True, 
                                label='▶️ Autoplay Audio'
                            )

                    with gr.Column():
                        # Audio output
                        audio_output = gr.Audio(
                            interactive=False, 
                            label='🔊 Output Audio', 
                            autoplay=True
                        )
                        
                        # Status message
                        status_box = gr.Textbox(
                            label="📡 Status",
                            interactive=False,
                            placeholder="Status will appear here..."
                        )
                        
                        # File downloads
                        audio_file_download = gr.File(
                            label='📥 Download Audio',
                            elem_classes="small-file-box"
                        )
                        
                        with gr.Accordion('📁 Additional Files', open=False):
                            word_level_srt_file = gr.File(
                                label='📝 Word-Level SRT',
                                elem_classes="small-file-box"
                            )
                            sentence_srt_file = gr.File(
                                label='📜 Sentence-Level SRT',
                                elem_classes="small-file-box"
                            )
                            json_file_download = gr.File(
                                label='⏳ Sentence Timestamp JSON',
                                elem_classes="small-file-box"
                            )
                        
                        # Autoplay toggle
                        autoplay_checkbox.change(
                            fn=toggle_autoplay,
                            inputs=[autoplay_checkbox],
                            outputs=[audio_output]
                        )

                # Connect generate button
                inputs_list = [
                    text_input, 
                    language_name, 
                    voice_name, 
                    speed, 
                    translate_text, 
                    remove_silence, 
                    keep_silence_up_to,
                    ip_check_username_input
                ]
                
                outputs_list = [
                    audio_output, 
                    audio_file_download,
                    word_level_srt_file,
                    sentence_srt_file,
                    json_file_download,
                    status_box
                ]
                
                text_input.submit(
                    KOKORO_TTS_API, 
                    inputs=inputs_list, 
                    outputs=outputs_list
                )
                
                generate_btn.click(
                    KOKORO_TTS_API, 
                    inputs=inputs_list, 
                    outputs=outputs_list
                )

        return demo

def tutorial():
    """Tutorial/guide interface."""
    explanation = """
    ## Language Code Explanation:
    Example: `'af_bella'` 
    - **'a'** stands for **American English**.
    - **'f_'** stands for **Female** (If it were 'm_', it would mean Male).
    - **'bella'** refers to the specific voice.

    The first character in the voice code stands for the language:
    - **"a"**: American English
    - **"b"**: British English
    - **"h"**: Hindi
    - **"e"**: Spanish
    - **"f"**: French
    - **"i"**: Italian
    - **"p"**: Brazilian Portuguese
    - **"j"**: Japanese
    - **"z"**: Mandarin Chinese

    The second character stands for gender:
    - **"f_"**: Female
    - **"m_"**: Male
    """
    
    with gr.Blocks() as demo2:
        gr.Markdown(explanation)
    
    return demo2

# ====================================================================
# === MAIN LAUNCH FUNCTION ===
# ====================================================================
import click

@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing.")
def main(debug, share):
    """Main launch function."""
    global pipeline, last_used_language, temp_folder
    
    # Initialize pipeline and temp folder
    pipeline = KPipeline(lang_code="a")
    last_used_language = "a"
    temp_folder = create_audio_dir()
    
    # Create interfaces
    demo1 = ui()
    demo2 = tutorial()
    
    # Create tabbed interface
    demo = gr.TabbedInterface(
        [demo1, demo2],
        ["Text To Speech", "Voice Character Guide"],
        title="Long Touch Generator 03060914996",
        css=css_hider
    )
    
    # Launch with authentication
    demo.queue().launch(
        debug=debug, 
        share=share, 
        show_api=False, 
        auth=custom_auth
    )

# ====================================================================
# === INITIALIZATION ===
# ====================================================================
if __name__ == "__main__":
    main()
