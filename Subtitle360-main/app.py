import datetime
import os
from fastapi import Request
from kokoro import KPipeline
from huggingface_hub import list_repo_files
import uuid
import re
import gradio as gr
import string
import json
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil
import click

# === SECURITY & RENTAL SETTINGS (Aapko sirf yeh section badalna hai) ===
# 1. IP File: Is file mein sirf allowed IP address save hoga
IP_FILE = "allowed_ip.txt" 

# 2. Authentication: Har Naye Client ke liye naya username aur password set karein
CLIENT_USERNAME = "rental_client_1" 
CLIENT_PASSWORD = "StrongPassword456"

# 3. Expiration: Tareekh ko is format mein set karein: Din-Mahina-Saal
EXPIRY_DATE_STR = "25-12-2025" 
# ======================================================================

css_hider = """
footer { visibility: hidden !important; height: 0px !important; }
.gradio-container { min-height: 0px !important; }
"""

# Yeh function har login ki koshish par chalega
def custom_auth(username, password, request: Request):
    client_ip = request.headers.get("x-forwarded-for") or request.client.host

    if username == CLIENT_USERNAME and password == CLIENT_PASSWORD:
        
        if not os.path.exists(IP_FILE):
            try:
                with open(IP_FILE, "w") as f:
                    f.write(client_ip)
                return True
            except Exception:
                return False
        
        else:
            try:
                with open(IP_FILE, "r") as f:
                    allowed_ip = f.read().strip()
            except Exception:
                return False

            if client_ip == allowed_ip:
                return True
            else:
                return False 
    
    return False


# translate langauge
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
    from deep_translator import GoogleTranslator
    lang_code=language_map_local[target_language]
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
    new_lang = language_map.get(Language, "a")

    if new_lang != last_used_language:
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang
        except Exception:
            pipeline = KPipeline(lang_code="a")
            last_used_language = "a"



def get_voice_names(repo_id):
    """Fetches and returns a list of voice names (without extensions) from the given Hugging Face repository."""
    return [os.path.splitext(file.replace("voices/", ""))[0] for file in list_repo_files(repo_id) if file.startswith("voices/")]

def create_audio_dir():
    """Creates the 'kokoro_audio' directory in the root folder if it doesn't exist."""
    root_dir = os.getcwd()
    audio_dir = os.path.join(root_dir, "kokoro_audio")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    return audio_dir

def clean_text(text):
    replacements = {
        "–": " ",
        "-": " ",
        "**": " ",
        "*": " ",
        "#": " ",
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
        r'[\U0001F1E0-\U0001F1FF]'
        r'', flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tts_file_name(text,language):
    global temp_folder
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = text.replace(" ", "_")
    language=language.replace(" ", "_").strip()
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else language
    random_string = uuid.uuid4().hex[:8].upper()
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name

def remove_silence_function(file_path,minimum_silence=50):
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                     min_silence_len=100,
                                     silence_thresh=-45,
                                     keep_silence=minimum_silence)
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
    timestamps={}
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        for i, result in enumerate(generator):
            gs = result.graphemes
            audio = result.audio
            tokens = result.tokens
            timestamps[i]={"text":gs,"words":[]}
            if Language in ["American English", "British English"]:
                for t in tokens:
                    timestamps[i]["words"].append({"word":t.text,"start":t.start_ts,"end":t.end_ts})
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
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


def write_word_srt(word_level_timestamps, output_file="word.srt", skip_punctuation=True):
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
    subtitles = []
    subtitle_words = []
    start_time = None

    remove_punctuation = ['"',"—"]

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
            text=split_line_by_char_limit(text, max_chars=30)
            f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")


def fix_punctuation(text):
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
    text=''.join(result)
    return text.strip()


def make_json(word_timestamps, json_file_name):
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


def modify_filename(save_path: str, prefix: str = ""):
    directory, filename = os.path.split(save_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{prefix}{name}{ext}"
    return os.path.join(directory, new_filename)

def save_current_data():
    if os.path.exists("./last"):
        shutil.rmtree("./last")
    os.makedirs("./last",exist_ok=True)
    
def KOKORO_TTS_API(text, Language="American English",voice="af_bella", speed=1,translate_text=False,remove_silence=False,keep_silence_up_to=0.05):
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
    
    

def ui():
    def toggle_autoplay(autoplay):
        return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

    
    
    with gr.Blocks() as demo:
        lang_list = ['American English', 'British English', 'Hindi', 'Spanish', 'French', 'Italian', 'Brazilian Portuguese', 'Japanese', 'Mandarin Chinese']
        voice_names = get_voice_names("hexgrad/Kokoro-82M")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label='📝 Enter Text', lines=3)
                
                with gr.Row():
                    language_name = gr.Dropdown(lang_list, label="🌍 Select Language", value=lang_list[0])

                with gr.Row():
                    voice_name = gr.Dropdown(voice_names, label="🎙️ Choose VoicePack", value='af_heart')

                with gr.Row():
                    generate_btn = gr.Button('🚀 Generate', variant='primary')

                with gr.Accordion('🎛️ Audio Settings', open=False):
                    speed = gr.Slider(minimum=0.25, maximum=2, value=1, step=0.1, label='⚡️Speed', info='Adjust the speaking speed')
                    translate_text = gr.Checkbox(value=False, label='🌐 Translate Text to Selected Language')
                    remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence ')

            with gr.Column():
                audio = gr.Audio(interactive=False, label='🔊 Output Audio', autoplay=True)
                audio_file = gr.File(label='📥 Download Audio')
                with gr.Accordion('🎬 Autoplay, Subtitle, Timestamp', open=False):
                    autoplay = gr.Checkbox(value=True, label='▶️ Autoplay')
                    autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])
                    word_level_srt_file = gr.File(label='📝 Download Word-Level SRT')
                    srt_file = gr.File(label='📜 Download Sentence-Level SRT')
                    sentence_duration_file = gr.File(label='⏳ Download Sentence Timestamp JSON')

        text.submit(KOKORO_TTS_API, inputs=[text, language_name, voice_name, speed,translate_text, remove_silence], outputs=[audio, audio_file,word_level_srt_file,srt_file,sentence_duration_file])
        generate_btn.click(KOKORO_TTS_API, inputs=[text, language_name, voice_name, speed,translate_text, remove_silence], outputs=[audio, audio_file,word_level_srt_file,srt_file,sentence_duration_file])
    return demo

def tutorial():
    explanation = """
    ## Language Code Explanation:
    Example: 'af_bella'  
    - 'a' stands for American English.
    - 'f_' stands for Female (If it were 'm_', it would mean Male).
    - 'bella' refers to the specific voice.

    The first character in the voice code stands for the language:
    - "a": American English
    - "b": British English
    - "h": Hindi
    - "e": Spanish
    - "f": French
    - "i": Italian
    - "p": Brazilian Portuguese
    - "j": Japanese
    - "z": Mandarin Chinese

    The second character stands for gender:
    - "f_": Female
    - "m_": Male
    """
    with gr.Blocks() as demo2:
        gr.Markdown(explanation)
    return demo2


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    # --- Date Check Logic ---
    try:
        EXPIRY_DATE = datetime.datetime.strptime(EXPIRY_DATE_STR, "%d-%m-%Y").date()
        CURRENT_DATE = datetime.date.today()
    except ValueError:
        EXPIRY_DATE = CURRENT_DATE = datetime.date.today()

    if CURRENT_DATE > EXPIRY_DATE:
        expired_interface = gr.Interface(
            fn=lambda: "Tool ki muddat (validity) khatam ho chuki hai. Mazeed istimaal ke liye rabta karein.", 
            inputs=None, 
            outputs="text",
            title="TOOL EXPIRED - ACCESS DENIED",
            allow_flagging="never"
        )
        expired_interface.launch(show_api=False)
        return
    
    # --- Launch Tool with Custom Authentication (IP Check included) ---
    demo1 = ui()
    demo2 = tutorial()
    
    demo = gr.TabbedInterface([demo1, demo2],["Text To Speech","Voice Character Guide"],title="Long Touch Generator 03060914996", css=css_hider)
    
    demo.queue().launch(debug=debug, share=share, show_api=False, auth=custom_auth)


# Initialize default pipeline
last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = create_audio_dir()
if __name__ == "__main__":
    main()
