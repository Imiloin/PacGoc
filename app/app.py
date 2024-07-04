from pacgoc.cls import CLS
import argparse
import pandas as pd
import gradio as gr
import os
import threading
import time
import sys
import wave

source = None
SAMPLING_RATE = 48000
isint16 = False


# Use Ctrl+C to quit the program
def signal_handler(sig, frame):
    print("Exiting...")
    sys.exit(0)


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


def gen_result():
    global cls_res
    while True:
        audio_len = source.get_queue_size()
        print(audio_len)
        if audio_len > 10:
            audio_data = source.get_queue_data()
            cls_res = cls(audio_data)
            print(cls_res)
        time.sleep(1)


def get_cls_result():
    global cls_res
    return pd.DataFrame(cls_res, columns=["Category", "Score"])


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
cls = CLS(sr=SAMPLING_RATE, isint16=isint16)
cls_res = []

# Start the result generator
th_result = threading.Thread(target=gen_result, daemon=True)
th_result.start()

with gr.Blocks() as demo:
    cls_result = gr.Dataframe()
    demo.load(
        get_cls_result, inputs=None, outputs=cls_result, every=1, show_progress=False
    )

demo.launch()
