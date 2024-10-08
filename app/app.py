import os
import re
import sys
import shutil
from pathlib import Path
import numpy as np

# add current directory to sys.path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# -----------------------------------------------------------------------------
# Generate and update config_user.py
# -----------------------------------------------------------------------------

config_path = current_dir / "config.py"
config_user_path = current_dir / "config_user.py"


def extract_vars(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # split the content into logical lines
    logical_lines = []
    current_line = ""
    for line in content.splitlines():
        if line.endswith("\\"):
            current_line += line[:-1].rstrip() + " "
        else:
            current_line += line
            logical_lines.append(current_line.strip())
            current_line = ""

    # regular expression pattern to match variable definitions
    var_pattern = re.compile(r"^(\w+)\s*=\s*(.*)")

    vars_set = {
        match.group(1) for line in logical_lines for match in var_pattern.finditer(line)
    }

    return vars_set


# check if config_user.py exists
if not config_user_path.exists():
    shutil.copy(config_path, config_user_path)
else:
    config_vars = extract_vars(config_path)
    user_config_vars = extract_vars(config_user_path)

    missing_vars = config_vars - user_config_vars

    # update config_user.py with missing variables from config.py
    if missing_vars:
        print(f"Adding {len(missing_vars)} missing variables to {config_user_path}")
        # read config.py
        with open(config_path, "r") as f:
            config_content = f.readlines()

        with open(config_user_path, "a") as f_user:
            for var in missing_vars:
                writing = False
                # track open brackets to check if a variable is defined in a single line
                open_brackets = 0
                for line in config_content:
                    # check if the variable is defined in the current line
                    if re.match(rf"^\s*{var}\s*=", line):
                        writing = True
                    if writing:
                        # update brackets count
                        open_brackets += (
                            line.count("(") + line.count("[") + line.count("{")
                        )
                        open_brackets -= (
                            line.count(")") + line.count("]") + line.count("}")
                        )
                        f_user.write(line)
                        # if current line is the end of the variable definition, stop writing
                        if (
                            open_brackets == 0
                            and not line.strip().endswith("\\")
                            and not line.strip().endswith("(")
                            and not line.strip().endswith(",")
                        ):
                            break

import config_user

# -----------------------------------------------------------------------------
# Import packages and basic setup
# -----------------------------------------------------------------------------

from pacgoc.cls import CLS
from pacgoc.profiling import AgeGender
from pacgoc.profiling import Emotion
from pacgoc.sv import Vector
from pacgoc.spoof import SpoofDetector
from pacgoc.asr import ASR
from pacgoc.ans import ANS
from pacgoc.separation import SourceSeparation
from llm import Qwen2

import argparse
import pandas as pd
import gradio as gr
import threading
import time
import signal

INTERVAL = config_user.INTERVAL
MAX_AUDIO_LEN = config_user.MAX_AUDIO_LEN
assert INTERVAL >= 2 and INTERVAL <= 20, "Interval should be between 2 and 20 seconds"
assert MAX_AUDIO_LEN > 20, "Max audio length should be greater than 20 seconds"

source = None
SAMPLING_RATE = 48000
isint16 = False
du = os.path.join(current_dir, "..", "wave", "du.wav")

# -----------------------------------------------------------------------------
# Hardware Controller
# -----------------------------------------------------------------------------

from pacgoc.serial_api import Serial

serial = None

if config_user.HARDWARE_CONTROLLER_ON:
    serial = Serial(
        baudrate=config_user.BAUD_RATE,
        bytesize=config_user.BYTESIZE,
        parity=config_user.PARITY,
        stopbits=config_user.STOPBITS,
    )


def send_command(command: str, max_retries: int = 3, timeout: float = 0.5) -> bool:
    """
    Send a command to the hardware controller and wait for a response.
    """
    global serial
    if config_user.HARDWARE_CONTROLLER_ON:
        for _ in range(max_retries):
            serial.write(command)
            response = serial.read(timeout=timeout)
            if response == config_user.SUCCESS:
                return True
    return False


def out_on():
    print("Output processed audio.")
    state = send_command(config_user.OUT_ON)
    if not state:
        gr.Info("Failed to turn on the output.")


def out_off():
    print("Output unprocessed audio.")
    state = send_command(config_user.OUT_OFF)
    if not state:
        gr.Info("Failed to turn off the output.")


def nc_on():
    print("Turning on the Noise Cancellation.")
    state = send_command(config_user.NC_ON)
    if not state:
        gr.Info("Failed to turn on the Noise Cancellation.")


def nc_off():
    print("Turning off the Noise Cancellation.")
    state = send_command(config_user.NC_OFF)
    if not state:
        gr.Info("Failed to turn off the Noise Cancellation.")


def nc_update():
    print("Updating Noise Cancellation parameters.")
    state = send_command(config_user.NC_UPDATE)
    if not state:
        gr.Info("Failed to update Noise Cancellation parameters.")


def aec_on():
    print("Turning on the Acoustic Echo Cancellation.")
    state = send_command(config_user.AEC_ON)
    if not state:
        gr.Info("Failed to turn on the Acoustic Echo Cancellation.")


def aec_off():
    print("Turning off the Acoustic Echo Cancellation.")
    state = send_command(config_user.AEC_OFF)
    if not state:
        gr.Info("Failed to turn off the Acoustic Echo Cancellation.")


def tm_up():
    print("Tone Modificaton pitch up.")
    state = send_command(config_user.TM_UP)
    if not state:
        gr.Info("Failed to tone modification pitch up.")


def tm_down():
    print("Tone Modificaton pitch down.")
    state = send_command(config_user.TM_DOWN)
    if not state:
        gr.Info("Failed to tone modification pitch down.")


def tm_off():
    print("Turning off the Tone Modificaton.")
    state = send_command(config_user.TM_OFF)
    if not state:
        gr.Info("Failed to turn off the Tone Modificaton.")


def record_start():
    print("Start recording.")
    state = send_command(config_user.RECORD_START)
    if not state:
        gr.Info("Failed to start recording.")


def record_end():
    print("End recording.")
    state = send_command(config_user.RECORD_END)
    if not state:
        gr.Info("Failed to end recording.")


def playback_start():
    print("Start playback.")
    state = send_command(config_user.PLAYBACK_START)
    if not state:
        gr.Info("Failed to start playback.")


def playback_end():
    print("End playback.")
    state = send_command(config_user.PLAYBACK_END)
    if not state:
        gr.Info("Failed to end playback.")


# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------

from pacgoc.utils.format import pcm32to16
from utils import is_valid_wav, write_wav


# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# -----------------------------------------------------------------------------
# PacGoc
# -----------------------------------------------------------------------------

from datetime import datetime

LISTENING = False
RECORDING = False
LISTENING_INFERENCE = False

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
    global LISTENING, LISTENING_INFERENCE
    global RECORDING
    if LISTENING:
        print("End listening")
        LISTENING = False
        LISTENING_INFERENCE = True
        audio_data = source.get_queue_data()
        inference(audio_data)
        LISTENING_INFERENCE = False
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


def open_cls():
    if config_user.AUDIO_CLASSIFICATION_ON:
        cls_checkbox(True)
    else:
        gr.Info("Audio classification is not enabled.")


def close_cls():
    if config_user.AUDIO_CLASSIFICATION_ON:
        cls_checkbox(False)
    else:
        gr.Info("Audio classification is not enabled.")


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
# Acoustic Noise Suppression
# -----------------------------------------------------------------------------

ans_on = False
ans_noisy = du
ans_res = du


def ans_checkbox(enable_ans):
    global ans_on
    ans_on = enable_ans


def get_ans_noisy():
    global ans_noisy
    return ans_noisy


def get_ans_result():
    global ans_res
    return ans_res


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
# LLM Chat
# -----------------------------------------------------------------------------

import re


def _extract_command(response: str):
    # Extract the command part using regex
    match = re.search(r"<command><\|(.+?)\|><\|(.+?)\|></command>", response)
    if not match:
        return None
    command = (match.group(1), match.group(2))
    print(f"Detected command: {command}")
    return command


execution_dict = {
    ("开启", "音频降噪"): nc_on,
    ("关闭", "音频降噪"): nc_off,
    ("开启", "回声消除"): aec_on,
    ("关闭", "回声消除"): aec_off,
    ("开启", "变声"): tm_up,
    ("关闭", "变声"): tm_off,
    ("开启", "音频分类"): open_cls,
    ("关闭", "音频分类"): close_cls,
}


def _execute_command(command: tuple[str]):
    # Execute the command
    if command[0] not in ["开启", "关闭"]:
        return None, None
    if command[1] not in ["音频降噪", "回声消除", "变声", "音频分类"]:
        return None, None
    print(f"Executing command: {command}")
    func = execution_dict[command]
    func()
    return command[0], command[1]


def process_command(response: str):
    command = _extract_command(response)
    if command:
        _execute_command(command)


def predict(_query, _chatbot, _task_history):
    global llm
    print(f"User: {_query}")
    _chatbot.append((_query, ""))
    full_response = ""
    response = ""
    for new_text in llm.chat_stream(
        _query,
        history=_task_history,
    ):
        response += new_text
        _chatbot[-1] = (_query, response)

        yield _chatbot
        full_response = response

    print(f"History: {_task_history}")
    _task_history.append((_query, full_response))
    print(f"Qwen2-Instruct: {full_response}")
    # Process command if any
    thread = threading.Thread(target=process_command, args=(full_response,))
    thread.start()


def regenerate(_chatbot, _task_history):
    global llm
    if not _task_history:
        yield _chatbot
        return
    item = _task_history.pop(-1)
    _chatbot.pop(-1)
    yield from predict(item[0], _chatbot, _task_history)


def reset_user_input():
    return gr.update(value="")


def reset_state(_chatbot, _task_history):
    global llm
    _task_history.clear()
    _chatbot.clear()
    llm.gc()
    return _chatbot


def start_talk():
    global asr
    _ = source.get_queue_data()  # flush the audio buffer
    if asr is not None:
        asr.clear_cache()
    print("Talking...")


def end_talk(_chatbot, _task_history):
    global asr
    audio_data = source.get_queue_data()
    prompt = asr(audio_data)
    yield from predict(prompt, _chatbot, _task_history)


# -----------------------------------------------------------------------------
# Main Function that Generates Results
# -----------------------------------------------------------------------------

MAX_INFER_LEN = MAX_AUDIO_LEN * SAMPLING_RATE


def inference(audio_data: np.ndarray):
    global LISTENING, LISTENING_INFERENCE
    global cls_on, cls_res
    global profile_on, profile_res
    global verify_on, verify_res
    global spoof_on, spoof_res
    global asr_on, asr_res
    global ans_on, ans_noisy, ans_res
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
        if not LISTENING_INFERENCE and len(res) > 1:
            res = res[:-1]  # remove the last punctuation
        print(res)
        asr_res = asr_res + "\n" + res
    else:
        asr_res = ""
    # acoustic noise suppression
    if ans_on:
        ans(audio_data)
        ans_noisy = os.path.join(config_user.ans_output_dir, "noisy.wav")
        ans_res = os.path.join(
            config_user.ans_output_dir, config_user.ans_output_filename
        )
    # audio source separation
    if separation_on:
        separation(audio_data)
        separation_mixture = os.path.join(
            config_user.separation_output_dir, "mixture.wav"
        )
        separation_res = os.path.join(
            config_user.separation_output_dir, config_user.separation_output_filename
        )


# infiniate loop to generate results
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
                gr.Info("Max audio length reached, auto end listening.")
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
spoof_detector = None
asr = None
ans = None
separation = None
llm = None

if config_user.AUDIO_CLASSIFICATION_ON:
    cls = CLS(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.cls_model_root,
        topk=3,
    )
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
        model_root=config_user.sv_model_root,
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
if config_user.AUTOMATIC_SPEECH_RECOGNITION_ON or config_user.LLM_CHAT_ON:
    asr = ASR(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.asr_model_root,
    )
if config_user.ACOUSTIC_NOISE_SUPPRESSION_ON:
    ans = ANS(
        sr=SAMPLING_RATE,
        isint16=isint16,
        model_root=config_user.ans_model_root,
        output_path=config_user.ans_output_dir,
        output_filename=config_user.ans_output_filename,
    )
if config_user.AUDIO_SOURCE_SEPARATION_ON:
    separation = SourceSeparation(
        sr=SAMPLING_RATE,
        query_sr=config_user.query_sr,
        isint16=isint16,
        ckpt=config_user.separation_ckpt,
        resume_ckpt=config_user.separation_resume_ckpt,
        query_folder=config_user.query_folder,
        output_path=config_user.separation_output_dir,
        output_filename=config_user.separation_output_filename,
    )
if config_user.LLM_CHAT_ON:
    llm = Qwen2(
        checkpoint_path=config_user.llm_model_root,
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
css = os.path.join(current_dir, "style.css")


with gr.Blocks(css=css) as demo:
    gr.Markdown("# PacGoc Control Panel")
    with gr.Tab("PacGoc"):
        interval = gr.Slider(
            minimum=2,
            maximum=20,
            step=1,
            value=config_user.INTERVAL,
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
    if config_user.HARDWARE_CONTROLLER_ON:
        with gr.Tab("硬件控制"):
            gr.Markdown("## 硬件控制")
            with gr.Accordion("Output Control", open=True):
                gr.Markdown("### 输出控制")
                with gr.Row():
                    out_on_btn = gr.Button("Processed", elem_id="seafoam")
                    out_off_btn = gr.Button("Unprocessed")
                    out_on_btn.click(
                        out_on,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    out_off_btn.click(
                        out_off,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
            with gr.Accordion("Noise Cancellation", open=True):
                gr.Markdown("### 音频降噪")
                with gr.Row():
                    nc_on_btn = gr.Button("Turn On", elem_id="seafoam")
                    nc_off_btn = gr.Button("Turn Off")
                    nc_on_btn.click(
                        nc_on,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    nc_off_btn.click(
                        nc_off,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                with gr.Row():
                    nc_update_btn = gr.Button("Update Parameters", elem_id="skyblue")
                    nc_update_btn.click(
                        nc_update,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
            with gr.Accordion("Acoustic Echo Cancellation", open=True):
                gr.Markdown("### 回声消除")
                with gr.Row():
                    aec_on_btn = gr.Button("Turn On", elem_id="seafoam")
                    aec_off_btn = gr.Button("Turn Off")
                    aec_on_btn.click(
                        aec_on,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    aec_off_btn.click(
                        aec_off,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
            with gr.Accordion("Tune Modification", open=True):
                gr.Markdown("### 人声调整")
                with gr.Row():
                    tm_up_btn = gr.Button("Pitch Up", elem_id="skyblue")
                    tm_down_btn = gr.Button("Pitch Down", elem_id="mistyrose")
                    tm_up_btn.click(
                        tm_up,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    tm_down_btn.click(
                        tm_down,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                with gr.Row():
                    tm_off_btn = gr.Button("Turn Off")
                    tm_off_btn.click(
                        tm_off,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
            with gr.Accordion("Record and Playback", open=True):
                gr.Markdown("### 录音和播放")
                with gr.Row():
                    record_start_btn = gr.Button("Start Recording", elem_id="skyblue")
                    record_end_btn = gr.Button("End Recording")
                    record_start_btn.click(
                        record_start,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    record_end_btn.click(
                        record_end,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                with gr.Row():
                    playback_start_btn = gr.Button(
                        "Start Playback", elem_id="mistyrose"
                    )
                    playback_end_btn = gr.Button("End Playback")
                    playback_start_btn.click(
                        playback_start,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    playback_end_btn.click(
                        playback_end,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
    if config_user.AUDIO_CLASSIFICATION_ON:
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
            if config_user.HARDWARE_CONTROLLER_ON:
                with gr.Row():
                    tm_up_btn = gr.Button("Pitch Up", elem_id="skyblue")
                    tm_down_btn = gr.Button("Pitch Down", elem_id="mistyrose")
                    tm_up_btn.click(
                        tm_up,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                    tm_down_btn.click(
                        tm_down,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
                    )
                with gr.Row():
                    tm_off_btn = gr.Button("Turn Off")
                    tm_off_btn.click(
                        tm_off,
                        inputs=None,
                        outputs=None,
                        show_progress="hidden",
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
    if config_user.ACOUSTIC_NOISE_SUPPRESSION_ON:
        with gr.Tab("音频去噪"):
            enable_ans = gr.Checkbox(
                value=False, label="Enable Acoustic Noise Suppression"
            )
            enable_ans.change(ans_checkbox, inputs=[enable_ans], outputs=None)
            ans_noisy = gr.Audio(
                label="noisy",
                type="filepath",
                every=1,
            )
            ans_result = gr.Audio(
                label="denoised",
                type="filepath",
                every=1,
            )
            demo.load(
                get_ans_noisy,
                inputs=None,
                outputs=ans_noisy,
                every=1,
                show_progress=False,
            )
            demo.load(
                get_ans_result,
                inputs=None,
                outputs=ans_result,
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
    if config_user.LLM_CHAT_ON:
        with gr.Tab("语言模型聊天"):
            chatbot = gr.Chatbot(label="PacGoc-Chat", elem_classes="control-height")
            with gr.Row():
                start_talk_btn = gr.Button("Start talking", scale=4)
                end_talk_btn = gr.Button("End talking", scale=4)
                retry_btn = gr.Button("🔄", scale=1)
            query = gr.Textbox(lines=2, label="Input")
            task_history = gr.State([])
            with gr.Row():
                empty_btn = gr.Button("🧹 Clear History")
                submit_btn = gr.Button("🚀 Submit")
                regen_btn = gr.Button("🤔️ Regenerate")

            submit_btn.click(
                predict,
                [query, chatbot, task_history],
                [chatbot],
                show_progress=True,
            )
            submit_btn.click(reset_user_input, inputs=[], outputs=[query])
            empty_btn.click(
                reset_state,
                [chatbot, task_history],
                outputs=[chatbot],
                show_progress=True,
            )
            regen_btn.click(
                regenerate,
                [chatbot, task_history],
                [chatbot],
                show_progress=True,
            )
            start_talk_btn.click(
                start_talk,
                inputs=None,
                outputs=None,
                show_progress="hidden",
            )
            end_talk_btn.click(
                end_talk,
                inputs=[chatbot, task_history],
                outputs=[chatbot],
                show_progress=True,
            )
            retry_btn.click(
                regenerate,
                [chatbot, task_history],
                [chatbot],
                show_progress=True,
            )


def share_auth(username, password):
    return username == config_user.username and password == config_user.password


# launch webui
if config_user.share:
    demo.launch(allowed_paths=[assets_dir], share=True, auth=share_auth)
else:
    demo.launch(allowed_paths=[assets_dir])
