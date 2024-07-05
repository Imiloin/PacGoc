import os
import sys
from pathlib import Path
import shutil

# add current directory to sys.path to import pcie module
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# -----------------------------------------------------------------------------
# Generate and update config_user.py
# -----------------------------------------------------------------------------

# 定义config.py和config_user.py的路径
current_dir = Path(__file__).parent
config_path = current_dir / "config.py"
config_user_path = current_dir / "config_user.py"

# 检查config_user.py是否存在
if not config_user_path.exists():
    # 如果不存在，则复制config.py为config_user.py
    shutil.copy(config_path, config_user_path)
else:
    # 如果存在，则检查config_user.py是否包含config.py中的所有条目
    with open(config_path, 'r') as f:
        config_vars = {line.split('=')[0].strip() for line in f if '=' in line}
    
    with open(config_user_path, 'r') as f:
        user_config_vars = {line.split('=')[0].strip() for line in f if '=' in line}
    
    missing_vars = config_vars - user_config_vars
    
    # 如果config_user.py缺少某些条目，则从config.py中复制这些条目
    if missing_vars:
        with open(config_path, 'r') as f:
            config_lines = f.readlines()
        
        with open(config_user_path, 'a') as f:
            for var in missing_vars:
                for line in config_lines:
                    if line.startswith(var):
                        f.write(line)
                        break

import config_user

from pacgoc.cls import CLS
from pacgoc.profiling import AgeGender
from pacgoc.profiling import Emotion
from pacgoc.verification import Vector
from pacgoc.asr import ASR
from pacgoc.separation import SourceSeparation
import argparse
import pandas as pd
import gradio as gr
import threading
import time
import signal
import wave

source = None
SAMPLING_RATE = 48000
isint16 = False
du = os.path.join(current_dir, "..", "wave", "du.wav")

import paddle

paddle.utils.run_check()


# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------


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
profile_res = []


def profile_checkbox(enable_profile):
    global profile_on
    profile_on = enable_profile


def get_profile_result():
    global profile_res
    return pd.DataFrame(profile_res, columns=["Gender", "Age", "Emotion"])


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
separation_res = du


def separation_checkbox(enable_separation):
    global separation_on
    separation_on = enable_separation


def get_separation_result():
    global separation_res
    return separation_res


# -----------------------------------------------------------------------------
# Main Function that Generates Results
# -----------------------------------------------------------------------------


def gen_result():
    global cls_on, cls_res
    global profile_on, profile_res
    global verify_on, verify_res
    global asr_on, asr_res
    global separation_on, separation_res, du
    # flush the audio buffer
    _ = source.get_queue_data()
    while True:
        audio_len = source.get_queue_size()
        print(audio_len)
        if audio_len > 10:
            audio_data = source.get_queue_data()
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
                profile_res = [
                    (
                        agegender_res["gender"],
                        agegender_res["age"],
                        emotion_res,
                    ),
                ]
                print(profile_res)
            else:
                profile_res = []
            # speaker verification
            if verify_on:
                verify_res = vector(audio_data)
                print(verify_res)
            else:
                verify_res = "Unknown"
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
                separation_res = os.path.join(config_user.output_dir, config_user.output_filename)
            else:
                separation_res = du
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

# Start the audio source
th_receive.start()

# Initialize the models
cls = None
age_gender = None
emotion = None
vector = None
asr = None
separation = None

cls = CLS(sr=SAMPLING_RATE, isint16=isint16)
age_gender = AgeGender(
    sr=SAMPLING_RATE,
    isint16=isint16,
    model_root=config_user.age_gender_model_root,
)
emotion = Emotion(sr=SAMPLING_RATE, isint16=isint16)
vector = Vector(
    sr=SAMPLING_RATE,
    isint16=isint16,
    enroll_embeddings=config_user.enroll_embeddings_json,
    enroll_audio_dir=config_user.enroll_audio_dir,
)
asr = ASR(
    sr=SAMPLING_RATE,
    isint16=isint16,
    model=config_user.asr_model_type,
)
separation = SourceSeparation(
    sr=SAMPLING_RATE,
    isint16=isint16,
    ckpt=config_user.ckpt,
    resume_ckpt=config_user.resume_ckpt,
    query_folder=config_user.query_folder,
    output_path=config_user.output_dir,
    output_filename=config_user.output_filename,
)

# Start the result generator
th_result = threading.Thread(target=gen_result, daemon=True)
th_result.start()


# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# PACGOC")
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
    with gr.Tab("音频人物画像"):
        gr.Markdown("## 音频人物画像")
        enable_profile = gr.Checkbox(value=False, label="Enable Speaker Profiling")
        enable_profile.change(profile_checkbox, inputs=[enable_profile], outputs=None)
        profile_result = gr.Dataframe()
        demo.load(
            get_profile_result,
            inputs=None,
            outputs=profile_result,
            every=1,
            show_progress=False,
        )
    with gr.Tab("声纹识别"):
        gr.Markdown("## 声纹识别")
        enable_verify = gr.Checkbox(value=False, label="Enable Speaker Verification")
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
    with gr.Tab("自动语音识别"):
        gr.Markdown("## 自动语音识别")
        enable_asr = gr.Checkbox(
            value=False, label="Enable Automatic Speech Recognition"
        )
        enable_asr.change(asr_checkbox, inputs=[enable_asr], outputs=None)
        asr_result = gr.TextArea(
            every=1,
        )
        demo.load(
            get_asr_result,
            inputs=None,
            outputs=asr_result,
            every=1,
            show_progress=False,
        )
    with gr.Tab("音乐人声分离"):
        gr.Markdown("## 音乐人声分离")
        enable_separation = gr.Checkbox(
            value=False, label="Enable Audio Source Separation"
        )
        enable_separation.change(
            separation_checkbox, inputs=[enable_separation], outputs=None
        )
        separation_result = gr.Audio(
            label="vocal",
            type="filepath",
            every=1,
        )
        demo.load(
            get_separation_result,
            inputs=None,
            outputs=separation_result,
            every=1,
            show_progress=False,
        )

demo.launch()
