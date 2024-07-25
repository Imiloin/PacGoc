import os
import sys
from pathlib import Path
import shutil
import numpy as np

# add current directory to sys.path to import pcie module
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# -----------------------------------------------------------------------------
# Generate and update config_user.py
# -----------------------------------------------------------------------------

config_path = current_dir / "config.py"
config_user_path = current_dir / "config_user.py"

# check if config_user.py exists
if not config_user_path.exists():
    shutil.copy(config_path, config_user_path)
else:
    # check if config_user.py is up-to-date with config.py
    with open(config_path, "r") as f:
        config_vars = {line.split("=")[0].strip() for line in f if "=" in line}

    with open(config_user_path, "r") as f:
        user_config_vars = {line.split("=")[0].strip() for line in f if "=" in line}

    missing_vars = config_vars - user_config_vars

    # update config_user.py with missing variables from config.py
    if missing_vars:
        with open(config_path, "r") as f:
            config_lines = f.readlines()

        with open(config_user_path, "a") as f:
            for var in missing_vars:
                for line in config_lines:
                    if line.startswith(var):
                        f.write(line)
                        break

import config_user

# -----------------------------------------------------------------------------
# Import packages and basic setup
# -----------------------------------------------------------------------------

from pacgoc.cls import CLS
from pacgoc.profiling import AgeGender
from pacgoc.profiling import Emotion
from pacgoc.verification import Vector
from pacgoc.spoof import SpoofDetector
from pacgoc.asr import ASR
from pacgoc.separation import SourceSeparation

import argparse
import pandas as pd
import gradio as gr
import threading
import time
import signal
import wave

INTERVAL = 10  # running interval, in seconds
MAX_AUDIO_LEN = 30  # maximum audio length, in seconds
source = None
SAMPLING_RATE = 48000
isint16 = False
du = os.path.join(current_dir, "..", "wave", "du.wav")

import paddle

paddle.utils.run_check()

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------

from pacgoc.utils.format import pcm32to16


# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def is_valid_wav(path):
    if not os.path.isfile(path):
        return False
    if not path.lower().endswith(".wav"):
        return False
    try:
        with wave.open(path, "rb") as f:
            return True
    except wave.Error:
        return False


def write_wav(data: np.ndarray, frame_rate: int, file_path: os.PathLike):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # pls make sure data is int16
    data = data.astype(np.int16)

    with wave.open(file_path, "w") as wav_file:
        n_channels = 1  # single channel
        sample_width = 2  # int16
        n_frames = len(data)

        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.setnframes(n_frames)

        # write data to file
        wav_file.writeframes(data.tobytes())


# -----------------------------------------------------------------------------
# PacGoc
# -----------------------------------------------------------------------------

from datetime import datetime

LISTENING = False
RECORDING = False

LISTENING_ON = "🔴 Listening"
LISTENING_OFF = "⚪ Listening"


def set_interval(interval):
    global INTERVAL
    INTERVAL = interval


def start_listen():
    global LISTENING
    global asr, asr_res
    LISTENING = True
    _ = source.get_queue_data()  # flush the audio buffer
    if config_user.AUTOMATIC_SPEECH_RECOGNITION_ON:
        # clear asr cache
        asr.clear_cache()
        asr_res = asr_res + "\n"
    print("Listening...")
    return LISTENING_ON


def end_listen():
    global LISTENING
    if LISTENING:
        print("End listening")
        LISTENING = False
        audio_data = source.get_queue_data()
        inference(audio_data)
        # if is recording, save the audio data to file
        if RECORDING:
            now = datetime.now()
            file_name = now.strftime("record-%Y-%m-%d-%H-%M-%S.wav")
            file_path = os.path.join(config_user.recordings_dir, file_name)
            if isint16:
                write_wav(audio_data, SAMPLING_RATE, file_path)
            else:
                write_wav(pcm32to16(audio_data), SAMPLING_RATE, file_path)
    return LISTENING_OFF


def get_listen_status():
    global LISTENING
    if LISTENING:
        return LISTENING_ON
    else:
        return LISTENING_OFF


def set_recording(recording):
    global RECORDING
    RECORDING = recording


# -----------------------------------------------------------------------------
# Audio Classification
# -----------------------------------------------------------------------------

cls_on = False
cls_res = []


def cls_checkbox(enable_cls):
    global cls_on
    cls_on = enable_cls


def get_cls_result():
    global cls_res
    return pd.DataFrame(cls_res, columns=["Category", "Score"])


# -----------------------------------------------------------------------------
# Speaker Profiling
# -----------------------------------------------------------------------------

profile_on = False
default_profile_res = {"result": [], "emojis": "😸"}
profile_res = default_profile_res


def get_genderage_emoji(age: int, gender: str) -> str:
    if gender == "男/Male":
        if age < 13:
            return "👶"
        elif age < 28:
            return "👦"
        elif age < 50:
            return "🧑"
        elif age < 70:
            return "🧔"
        else:
            return "👴"
    elif gender == "女/Female":
        if age < 13:
            return "👶"
        elif age < 28:
            return "👧"
        elif age < 50:
            return "👩"
        elif age < 70:
            return "👱"
        else:
            return "👵"
    else:
        print("Unknown gender", str(gender))
        return "🧑‍🦲"


emotion_dict = {
    "中立/neutral": "😐",
    "开心/happy": "😄",
    "生气/angry": "😠",
    "厌恶/disgusted": "😒",
    "恐惧/fearful": "😱",
    "难过/sad": "😔",
    "吃惊/surprised": "😲",
    "其他/other": "🤔",
    "<unk>": "😶",
}


def process_profile_result(agegender_res, emotion_res):
    gender = agegender_res["gender"]

    age = agegender_res["age"]
    genderage_emoji = get_genderage_emoji(age, gender)

    emotion = emotion_res
    emotion_emoji = emotion_dict[emotion]

    model_output = [
        (gender, str(age), emotion),
    ]
    emojis = genderage_emoji + " " + emotion_emoji

    profile_res = {"result": model_output, "emojis": emojis}

    return profile_res


def profile_checkbox(enable_profile):
    global profile_on
    profile_on = enable_profile


def get_profile_result():
    global profile_res
    return pd.DataFrame(profile_res["result"], columns=["Gender", "Age", "Emotion"])


def get_profile_emojis():
    global profile_res
    return profile_res["emojis"]


# -----------------------------------------------------------------------------
# Speaker Verification
# -----------------------------------------------------------------------------

verify_on = False
verify_res = "Unknown"


def verify_checkbox(enable_verify):
    global verify_on
    verify_on = enable_verify


def get_verify_result():
    global verify_res
    return [("Speaker: ", None), (verify_res, verify_res)]


# -----------------------------------------------------------------------------
# Spoof Detection
# -----------------------------------------------------------------------------

spoof_on = False
spoof_res = "bonafide"


def spoof_checkbox(enable_spoof):
    global spoof_on
    spoof_on = enable_spoof


def get_spoof_result():
    global spoof_res
    return [("Result: ", None), (spoof_res, spoof_res)]


# -----------------------------------------------------------------------------
# Automatic Speech Recognition
# -----------------------------------------------------------------------------

asr_on = False
asr_res = ""


def asr_checkbox(enable_asr):
    global asr_on
    asr_on = enable_asr


def get_asr_result():
    global asr_res
    return asr_res


# -----------------------------------------------------------------------------
# Audio Source Separation
# -----------------------------------------------------------------------------

separation_on = False
separation_mixture = du
separation_res = du


def separation_checkbox(enable_separation):
    global separation_on
    separation_on = enable_separation


def get_separation_mixture():
    global separation_mixture
    return separation_mixture


def get_separation_result():
    global separation_res
    return separation_res


# -----------------------------------------------------------------------------
# Main Function that Generates Results
# -----------------------------------------------------------------------------

MAX_INFER_LEN = MAX_AUDIO_LEN * SAMPLING_RATE


def inference(audio_data: np.ndarray):
    global cls_on, cls_res
    global profile_on, profile_res
    global verify_on, verify_res
    global spoof_on, spoof_res
    global asr_on, asr_res
    global separation_on, separation_mixture, separation_res
    if len(audio_data) > MAX_INFER_LEN:
        audio_data = audio_data[:MAX_INFER_LEN]
    # audio classification
    if cls_on:
        cls_res = cls(audio_data)
        print(cls_res)
    else:
        cls_res = []
    # speaker profiling
    if profile_on:
        agegender_res = age_gender(audio_data)
        emotion_res = emotion(audio_data)
        profile_res = process_profile_result(agegender_res, emotion_res)
        print(profile_res["result"])
    else:
        profile_res = default_profile_res
    # speaker verification
    if verify_on:
        verify_res = vector(audio_data)
        print(verify_res)
    else:
        verify_res = "Unknown"
    if spoof_on:
        spoof_res = spoof_detector(audio_data)
        print(spoof_res)
    # automatic speech recognition
    if asr_on:
        res = asr(audio_data)
        print(res)
        asr_res = asr_res + "\n" + res
    else:
        asr_res = ""
    # audio source separation
    if separation_on:
        separation(audio_data)
        separation_mixture = os.path.join(config_user.output_dir, "mixture.wav")
        separation_res = os.path.join(
            config_user.output_dir, config_user.output_filename
        )


def gen_result():
    global LISTENING
    # flush the audio buffer
    if source.get_queue_size():
        _ = source.get_queue_data()
    while True:
        audio_len = source.get_queue_size()
        print(audio_len)
        if LISTENING:
            if audio_len > MAX_AUDIO_LEN:
                print("Max audio length reached, auto end listening.")
                end_listen()
        else:
            if audio_len > INTERVAL:
                audio_data = source.get_queue_data()
                inference(audio_data)
        time.sleep(1)


# -----------------------------------------------------------------------------
# Program Entry Point
# -----------------------------------------------------------------------------

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="pcie")
args = parser.parse_args()

# Pick the source of audio data
if args.source == "pcie":
    print("Using PCIe source")
    # PCIe source uses 48kHz sampling rate
    SAMPLING_RATE = 48000
    isint16 = True
    from pacgoc.pcie_api import PCIe

    source = PCIe()
    th_receive = threading.Thread(target=source.receive, daemon=True)
elif args.source == "speaker":
    print("Using speaker source")
    isint16 = False
    from pacgoc.record import Recorder

    # SAMPLING_RATE can be changed
    source = Recorder(sr=SAMPLING_RATE)
    th_receive = threading.Thread(target=source.record, daemon=True)
elif is_valid_wav(args.source):
    print("Using WAV file source")
    from pacgoc.readwav import Wave

    source = Wave(args.source)
    SAMPLING_RATE = source.get_sample_rate()
    isint16 = True
    th_receive = threading.Thread(target=source.read, daemon=True)
else:
    print("Invalid source")
    print("Usage: python app.py --source [pcie|speaker|path/to/wav]")
    exit(1)


# Initialize the models
cls = None
age_gender = None
emotion = None
vector = None
asr = None
separation = None

if config_user.AUDIO_CLASSIFIER_ON:
    cls = CLS(sr=SAMPLING_RATE, isint16=isint16)
if config_user.SPEAKER_PROFILING_ON:
    age_gender = AgeGender(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.age_gender_model_root,
    )
    emotion = Emotion(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.emotion_model_root,
    )
if config_user.SPEAKER_VERIFICATION_ON:
    vector = Vector(
        sr=SAMPLING_RATE,
        isint16=isint16,
        threshold=0.6,  # threshold for verification
        enroll_embeddings=config_user.enroll_embeddings_json,
        enroll_audio_dir=config_user.enroll_audio_dir,
    )
if config_user.SPOOF_DETECTION_ON:
    spoof_detector = SpoofDetector(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.spoof_model_root,
    )
if config_user.AUTOMATIC_SPEECH_RECOGNITION_ON:
    asr = ASR(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.asr_model_root,
    )
if config_user.AUDIO_SOURCE_SEPARATION_ON:
    separation = SourceSeparation(
        sr=SAMPLING_RATE,
        query_sr=config_user.query_sr,
        isint16=isint16,
        ckpt=config_user.ckpt,
        resume_ckpt=config_user.resume_ckpt,
        query_folder=config_user.query_folder,
        output_path=config_user.output_dir,
        output_filename=config_user.output_filename,
    )

# Start the audio source
th_receive.start()

# Start the result generator
th_result = threading.Thread(target=gen_result, daemon=True)
th_result.start()


# -----------------------------------------------------------------------------
# Gradio webui
# -----------------------------------------------------------------------------

assets_dir = os.path.join(current_dir, "assets")
pacgoc_logo = os.path.join(assets_dir, "pacgoc.svg")

# remove loading orange border
# and add centering large elem_id
css = """
.generating {
    border: none;
}
#large span{
    font-size: 4em;
    text-align: center;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# PacGoc Control Panel")
    with gr.Tab("PacGoc"):
        interval = gr.Slider(
            minimum=2,
            maximum=20,
            step=1,
            value=10,
            label="Interval (seconds)",
            info="Recognized audio will be updated every interval seconds.",
        )
        with gr.Row(equal_height=False):
            start_btn = gr.Button("Start listening", scale=3)
            end_btn = gr.Button("End listening", scale=3)
            enable_recording = gr.Checkbox(
                value=False, label="Record Audio", scale=1, every=1
            )
            listen_status = gr.Textbox(
                LISTENING_OFF,
                container=False,
                show_label=False,
                interactive=False,
                scale=1,
            )
            start_btn.click(
                start_listen,
                inputs=None,
                outputs=listen_status,
                show_progress="hidden",
            )
            end_btn.click(
                end_listen,
                inputs=None,
                outputs=listen_status,
                show_progress="hidden",
            )
            enable_recording.change(
                set_recording, inputs=[enable_recording], outputs=None
            )
            demo.load(
                get_listen_status,
                inputs=None,
                outputs=listen_status,
                every=1,
                show_progress="hidden",
            )
        with gr.Row(equal_height=False):
            gr.HTML(
                f"""
            <div align="center">
                <img src=/file={pacgoc_logo} alt="logo" width="150"/>
            </div>
            """
            )
            interval.change(set_interval, inputs=[interval], outputs=None)
    if config_user.AUDIO_CLASSIFIER_ON:
        with gr.Tab("音频分类"):
            gr.Markdown("## 音频分类")
            enable_cls = gr.Checkbox(value=False, label="Enable Audio Classification")
            enable_cls.change(cls_checkbox, inputs=[enable_cls], outputs=None)
            cls_result = gr.Dataframe()
            demo.load(
                get_cls_result,
                inputs=None,
                outputs=cls_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )
    if config_user.SPEAKER_PROFILING_ON:
        with gr.Tab("音频人物画像"):
            gr.Markdown("## 音频人物画像")
            enable_profile = gr.Checkbox(value=False, label="Enable Speaker Profiling")
            enable_profile.change(
                profile_checkbox, inputs=[enable_profile], outputs=None
            )
            profile_result = gr.Dataframe()
            demo.load(
                get_profile_result,
                inputs=None,
                outputs=profile_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )
            with gr.Row(equal_height=False):
                profile_emojis = gr.Markdown("😸", elem_id="large")
                demo.load(
                    get_profile_emojis,
                    inputs=None,
                    outputs=profile_emojis,
                    every=1,
                    show_progress="hidden",
                )
    if config_user.SPEAKER_VERIFICATION_ON:
        with gr.Tab("声纹识别"):
            gr.Markdown("## 声纹识别")
            enable_verify = gr.Checkbox(
                value=False, label="Enable Speaker Verification"
            )
            enable_verify.change(verify_checkbox, inputs=[enable_verify], outputs=None)
            verify_result = gr.HighlightedText(
                show_legend=False,
                show_label=False,
                color_map=config_user.color_map,
            )
            demo.load(
                get_verify_result,
                inputs=None,
                outputs=verify_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )
    if config_user.SPOOF_DETECTION_ON:
        with gr.Tab("变声检测"):
            gr.Markdown("## 变声检测")
            enable_spoof = gr.Checkbox(value=False, label="Enable Spoof Detection")
            enable_spoof.change(spoof_checkbox, inputs=[enable_spoof], outputs=None)
            spoof_result = gr.HighlightedText(
                show_legend=False,
                show_label=False,
                color_map={"bonafide": "green", "spoof": "red"},
            )
            demo.load(
                get_spoof_result,
                inputs=None,
                outputs=spoof_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )
    if config_user.AUTOMATIC_SPEECH_RECOGNITION_ON:
        with gr.Tab("自动语音识别"):
            gr.Markdown("## 自动语音识别")
            enable_asr = gr.Checkbox(
                value=False, label="Enable Automatic Speech Recognition"
            )
            enable_asr.change(asr_checkbox, inputs=[enable_asr], outputs=None)
            asr_result = gr.TextArea(
                every=1,
                max_lines=10,
            )
            demo.load(
                get_asr_result,
                inputs=None,
                outputs=asr_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )
    if config_user.AUDIO_SOURCE_SEPARATION_ON:
        with gr.Tab("音乐人声分离"):
            gr.Markdown("## 音乐人声分离")
            enable_separation = gr.Checkbox(
                value=False, label="Enable Audio Source Separation"
            )
            enable_separation.change(
                separation_checkbox, inputs=[enable_separation], outputs=None
            )
            separation_mixture = gr.Audio(
                label="mixture",
                type="filepath",
                every=1,
            )
            separation_result = gr.Audio(
                label="vocal",
                type="filepath",
                every=1,
            )
            demo.load(
                get_separation_mixture,
                inputs=None,
                outputs=separation_mixture,
                every=1,
                show_progress=False,
            )
            demo.load(
                get_separation_result,
                inputs=None,
                outputs=separation_result,
                every=1,
                show_progress=False,
            )
            with gr.Row(equal_height=False):
                start_btn = gr.Button("Start listening", scale=3.5)
                end_btn = gr.Button("End listening", scale=3.5)
                listen_status = gr.Textbox(
                    LISTENING_OFF,
                    container=False,
                    show_label=False,
                    interactive=False,
                    scale=1,
                )
                start_btn.click(
                    start_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                end_btn.click(
                    end_listen,
                    inputs=None,
                    outputs=listen_status,
                    show_progress="hidden",
                )
                demo.load(
                    get_listen_status,
                    inputs=None,
                    outputs=listen_status,
                    every=1,
                    show_progress="hidden",
                )


def share_auth(username, password):
    return username == config_user.username and password == config_user.password


# launch webui
if config_user.share:
    demo.launch(allowed_paths=[assets_dir], share=True, auth=share_auth)
else:
    demo.launch(allowed_paths=[assets_dir])
