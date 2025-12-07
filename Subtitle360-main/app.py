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
    text = text.lower().strip()              # Convert to lowercase and strip leading/trailing spaces
    text = text.replace(" ", "_")            # Replace spaces with underscores
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
                millisec = int((seconds % 1) * 1000)
                return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"

            start_srt = format_srt_time(start_time)
            end_srt = format_srt_time(end_time)

            # Write entry to SRT file
            f.write(f"{index}\n{start_srt} --> {end_srt}\n{word}\n\n")
            index += 1  # Increment subtitle number

import string


def split_line_by_char_limit(text, max_chars=30):
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
        # Check if last line is a single word and there is a previous line
        if len(current_line.split()) == 1 and len(lines) > 0:
            # Append single word to previous line
            lines[-1] += " " + current_line
        else:
            lines.append(current_line)

    return "\n".join(lines)


def write_sentence_srt(word_level_timestamps, output_file="subtitles.srt", max_words=8, min_pause=0.1):
    subtitles = []  # Stores subtitle blocks
    subtitle_words = []  # Temporary list for words in the current subtitle
    start_time = None  # Tracks start time of current subtitle

    remove_punctuation = ['"',"—"]  # Add punctuations to remove if needed

    for i, entry in enumerate(word_level_timestamps):
        word = entry["word"]
        word_start = entry["start"]
        word_end = entry["end"]

        # Skip selected punctuation from remove_punctuation list
        if word in remove_punctuation:
            continue  

        # Attach punctuation to the previous word
        if word in string.punctuation:
            if subtitle_words:
                subtitle_words[-1] = (subtitle_words[-1][0] + word, subtitle_words[-1][1])
            continue  

        # Start a new subtitle block if needed
        if start_time is None:
            start_time = word_start

        # Calculate pause duration if this is not the first word
        if subtitle_words:
            last_word_end = subtitle_words[-1][1]
            pause_duration = word_start - last_word_end
        else:
            pause_duration = 0

        # **NEW FIX:** If pause is too long, create a new subtitle but ensure continuity
        if (word.endswith(('.', '!', '?')) and len(subtitle_words) >= 5) or len(subtitle_words) >= max_words or pause_duration > min_pause:
            end_time = subtitle_words[-1][1]  # Use last word's end time
            subtitle_text = " ".join(w[0] for w in subtitle_words)
            subtitles.append((start_time, end_time, subtitle_text))

            # Reset for the next subtitle, but **ensure continuity**
            subtitle_words = [(word, word_end)]  # **Carry the current word to avoid delay**
            start_time = word_start  # **Start at the current word, not None**

            continue  # Avoid adding the word twice

        # Add the current word to the subtitle
        subtitle_words.append((word, word_end))

    # Ensure last subtitle is added if anything remains
    if subtitle_words:
        end_time = subtitle_words[-1][1]
        subtitle_text = " ".join(w[0] for w in subtitle_words)
        subtitles.append((start_time, end_time, subtitle_text))

    # Function to format SRT timestamps
    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        millisec = int((seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{sec:02},{millisec:03}"

    # Write subtitles to SRT file
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, start=1):
            text=split_line_by_char_limit(text, max_chars=30)
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")

    # print(f"SRT file '{output_file}' created successfully!")


import json
import re

def fix_punctuation(text):
    # Remove spaces before punctuation marks (., ?, !, ,)
    text = re.sub(r'\s([.,?!])', r'\1', text)
    
    # Handle quotation marks: remove spaces before and after them
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = text.replace('" ', '"')
    
    # Track quotation marks to add space after closing quotes
    track = 0
    result = []
    
    for index, char in enumerate(text):
        if char == '"':
            track += 1
            result.append(char)
            # If it's a closing quote (even number of quotes), add a space after it
            if track % 2 == 0:
                result.append(' ')
        else:
            result.append(char)
    text=''.join(result)
    return text.strip()



def make_json(word_timestamps, json_file_name):
    data = {}
    temp = []
    inside_quote = False  # Track if we are inside a quoted sentence
    start_time = word_timestamps[0]['start']  # Initialize with the first word's start time
    end_time = word_timestamps[0]['end']  # Initialize with the first word's end time
    words_in_sentence = []
    sentence_id = 0  # Initialize sentence ID

    # Process each word in word_timestamps
    for i, word_data in enumerate(word_timestamps):
        word = word_data['word']
        word_start = word_data['start']
        word_end = word_data['end']

        # Collect word info for JSON
        words_in_sentence.append({'word': word, 'start': word_start, 'end': word_end})

        # Update the end_time for the sentence based on the current word
        end_time = word_end

        # Properly handle opening and closing quotation marks
        if word == '"':
            if inside_quote:
                temp[-1] += '"'  # Attach closing quote to the last word
            else:
                temp.append('"')  # Keep opening quote as a separate entry
            inside_quote = not inside_quote  # Toggle inside_quote state
        else:
            temp.append(word)

        # Check if this is a sentence-ending punctuation
        if word.endswith(('.', '?', '!')) and not inside_quote:
            # Ensure the next word is NOT a dialogue tag before finalizing the sentence
            if i + 1 < len(word_timestamps):
                next_word = word_timestamps[i + 1]['word']
                if next_word[0].islower():  # Likely a dialogue tag like "he said"
                    continue  # Do not break the sentence yet

            # Store the full sentence for JSON and reset word collection for next sentence
            sentence = " ".join(temp)
            sentence = fix_punctuation(sentence)  # Fix punctuation in the sentence
            data[sentence_id] = {
                'text': sentence,
                'duration': end_time - start_time,
                'start': start_time,
                'end': end_time,
                'words': words_in_sentence
            }

            # Reset for the next sentence
            temp = []
            words_in_sentence = []
            start_time = word_data['start']  # Update the start time for the next sentence
            sentence_id += 1  # Increment sentence ID

    # Handle any remaining words if necessary
    if temp:
        sentence = " ".join(temp)
        sentence = fix_punctuation(sentence)  # Fix punctuation in the sentence
        data[sentence_id] = {
            'text': sentence,
            'duration': end_time - start_time,
            'start': start_time,
            'end': end_time,
            'words': words_in_sentence
        }

    # Write data to JSON file
    with open(json_file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return json_file_name


def modify_filename(save_path: str, prefix: str = ""):
    directory, filename = os.path.split(save_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{prefix}{name}{ext}"
    return os.path.join(directory, new_filename)

def save_current_data():
    if os.path.exists("./last"):
        shutil.rmtree("./last")
    os.makedirs("./last",exist_ok=True)
    
# ====================================================================
# === 3. KOKORO_TTS_API Function (IP Lock Logic - Wohi hai) ===
# ==================================================================== 
def KOKORO_TTS_API(text, Language="American English",voice="af_bella", speed=1,translate_text=False,remove_silence=False,keep_silence_up_to=0.05, ip_check_username: str = None, request: gr.Request = None):
    # 'request: gr.Request = None' function signature mein hona hi error fix hai.

    # === IP/Security Check Start ===
    if not ip_check_username:
         gr.Warning("Access Denied: Please enter your Username in the 'User Name Enter' box.", duration=7)
         return None, None, None, None, None

    if request is None:
        gr.Warning("Error: Could not retrieve client information. Access denied.", duration=5)
        return None, None, None, None, None

    # IP address ko mazbooti se hasil karna
    client_ip = request.headers.get("x-forwarded-for", "UNKNOWN")
    if client_ip == "UNKNOWN":
        client_ip = request.client.host if request.client else "UNKNOWN"
    
    if client_ip == "UNKNOWN":
        gr.Warning("Error: IP address could not be determined. Access denied.", duration=5)
        return None, None, None, None, None
    
    # Step 1: Load the IP Map
    ip_map = load_ip_map()
    username = ip_check_username.strip().lower() # Username ko normalize karna
    
    if username in ip_map:
        # Step 2: User ka IP pehle se saved hai
        allowed_ip = ip_map[username]
        if client_ip != allowed_ip:
            # IP mismatch: Access Denied
            gr.Warning(f"Access Denied for user '{username}': This account is locked to IP {allowed_ip}. Your IP ({client_ip}) is different.", duration=10)
            return None, None, None, None, None
        else:
            # IP matched: Allow access
            gr.Info(f"IP check successful for user '{username}'.", duration=3)
    else:
        # Step 3: First time generation for this user. IP save karo.
        ip_map[username] = client_ip
        if save_ip_map(ip_map):
            gr.Info(f"First generation successful for user '{username}'. Your current IP ({client_ip}) is now locked to this account.", duration=7)
        else:
            gr.Warning("Warning: Could not save IP lock file. Continuing session, but IP lock may not work.", duration=5)

    # === IP/Security Check End ===
    
    # ORIGINAL TTS CODE CONTINUES
    if translate_text:        
        text=bulk_translate(text, Language, chunk_size=500)
    save_path,timestamps=generate_and_save_audio(text=text, Language=Language,voice=voice, speed=speed,remove_silence=remove_silence,keep_silence_up_to=keep_silence_up_to)
    if remove_silence==False:
        if Language in ["American English", "British English"]:
            word_level_timestamps=adjust_timestamps(timestamps)
            word_level_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="word_level_")
            normal_srt = modify_filename(save_path.replace(".wav", ".srt"), prefix="sentence_")
            json_file = modify_filename(save_path.replace(".wav", ".json"), prefix="duration_")
            write_word_srt(word_level_timestamps, output_file=word_level_srt, skip_punctuation=True)
            write_sentence_srt(word_level_timestamps, output_file=normal_srt, min_pause=0.01)
            make_json(word_level_timestamps, json_file)
            save_current_data()
            shutil.copy(save_path, "./last/")
            shutil.copy(word_level_srt, "./last/")
            shutil.copy(normal_srt, "./last/")
            shutil.copy(json_file, "./last/")
            return save_path,save_path,word_level_srt,normal_srt,json_file
    return save_path,save_path,None,None,None
# ==================================================================== 
    

def ui():
    def toggle_autoplay(autoplay):
        return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

    
    
    with gr.Blocks() as demo:
        # gr.Markdown("<center><h1 style='font-size: 40px;'>KOKORO TTS</h1></center>")  # Title
        # GitHub/Install Link yahan se hata diya gaya hai
        lang_list = ['American English', 'British English', 'Hindi', 'Spanish', 'French', 'Italian', 'Brazilian Portuguese', 'Japanese', 'Mandarin Chinese']
        voice_names = get_voice_names("hexgrad/Kokoro-82M")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label='📝 Enter Text', lines=3)
                
                with gr.Row():
                    # === Input for IP Lock (Must be entered by the user) ===
                    ip_check_username_input = gr.Textbox(
                        label='🔒 User Name Enter', 
                        placeholder='Enter the same username you used to login (e.g., user1, ali_tts).',
                        info='Security: Your account will be locked to the IP address used for the first successful generation with this username.'
                    )
                    # ==========================================================

                with gr.Row():
                    language_name = gr.Dropdown(lang_list, label="🌍 Select Language", value=lang_list[0])

                with gr.Row():
                    voice_name = gr.Dropdown(voice_names, label="🎙️ Choose Character", value='af_heart')#voice_names[0])

                with gr.Row():
                    generate_btn = gr.Button('🚀 Generate', variant='primary')

                with gr.Accordion('🎛️ Audio Settings', open=False):
                    speed = gr.Slider(minimum=0.25, maximum=2, value=1, step=0.1, label='⚡️Speed', info='Adjust the speaking speed')
                    translate_text = gr.Checkbox(value=False, label='🌐 Translate Text to Selected Language')
                    remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence ')
                    # === UI element for keep_silence_up_to ===
                    keep_silence_up_to = gr.Slider(minimum=0.01, maximum=0.5, value=0.05, step=0.01, label='Quiet Gap Size (Seconds)', info='How long of a silence gap to keep when removing silence.')
                    # ===================================================

            with gr.Column():
                audio = gr.Audio(interactive=False, label='🔊 Output Audio', autoplay=True)
                audio_file = gr.File(label='📥 Download Audio')
                with gr.Accordion('🎬 Autoplay, Subtitle, Timestamp', open=False):
                    autoplay = gr.Checkbox(value=True, label='▶️ Autoplay')
                    autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])
                    word_level_srt_file = gr.File(label='📝 Download Word-Level SRT')
                    srt_file = gr.File(label='📜 Download Sentence-Level SRT')
                    sentence_duration_file = gr.File(label='⏳ Download Sentence Timestamp JSON')

        # === UI Bindings Updated to include the new Username input (Wohi hai) ===
        inputs_list = [
            text, 
            language_name, 
            voice_name, 
            speed, 
            translate_text, 
            remove_silence, 
            keep_silence_up_to,
            ip_check_username_input # User must input this for IP check
        ]
        
        text.submit(KOKORO_TTS_API, inputs=inputs_list, outputs=[audio, audio_file,word_level_srt_file,srt_file,sentence_duration_file])
        generate_btn.click(KOKORO_TTS_API, inputs=inputs_list, outputs=[audio, audio_file,word_level_srt_file,srt_file,sentence_duration_file])
        # =====================================================================

        # Add examples to the interface
        

    return demo

def tutorial():
    # Markdown explanation for language code
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
        # gr.Markdown("[Install on Your Local System](https://github.com/NeuralFalconYT/kokoro_v1)") # Link yahan se bhi hata diya gaya hai
        gr.Markdown(explanation)  # Display the explanation
    return demo2



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    demo1 = ui()
    demo2 = tutorial()
    
    # CSS HIDER lagaya gaya hai takay footer aur links chhip jaaein
    demo = gr.TabbedInterface([demo1, demo2],["Text To Speech","Voice Character Guide"],title="Long Touch Generator 03060914996", css=css_hider)
    
    # ====================================================================
    # === 5. Launch Command (custom_auth multi-user login ke liye) ===
    # ====================================================================
    demo.queue().launch(
        debug=debug, 
        share=share, 
        show_api=False, 
        auth=custom_auth # custom_auth function ab multi-user login aur INDIVIDUAL expiry check karega
    )
    # ====================================================================



# Initialize default pipeline
last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = create_audio_dir()
if __name__ == "__main__":
    main()
