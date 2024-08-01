import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Recordings
# -----------------------------------------------------------------------------

recordings_dir = os.path.join(current_dir, "..", "recordings")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

share = True
username = "admin"
password = "admin"

INTERVAL = 10  # running interval, in seconds
MAX_AUDIO_LEN = 30  # maximum audio length, in seconds

# -----------------------------------------------------------------------------
# ON/OFF
# -----------------------------------------------------------------------------

HARDWARE_CONTROLLER_ON = True
AUDIO_CLASSIFIER_ON = True
SPEAKER_PROFILING_ON = True
SPEAKER_VERIFICATION_ON = True
SPOOF_DETECTION_ON = True
AUTOMATIC_SPEECH_RECOGNITION_ON = True
AUDIO_SOURCE_SEPARATION_ON = True

# -----------------------------------------------------------------------------
# Hardware Controller
# -----------------------------------------------------------------------------

# serial
BAUD_RATE = 115200
BYTESIZE = 8
PARITY = "E"
STOPBITS = 1

# return codes
SUCCESS = "\x00"
FAILURE = "\xff"

# output processed audio
OUT_ON = "PO"
OUT_OFF = "PF"

# noise cancellation
NC_ON = "NO"
NC_OFF = "NF"
NC_UPDATE = "NR"

# acoustic echo cancellation
AEC_ON = "AO"
AEC_OFF = "AF"

# tone modification
TM_UP = "TU"
TM_DOWN = "TD"
TM_OFF = "TF"

# -----------------------------------------------------------------------------
# Speaker Profiling
# -----------------------------------------------------------------------------

age_gender_model_root = os.path.join(
    current_dir, "..", "models", "wav2vec2-large-robust-24-ft-age-gender"
)
emotion_model_root = os.path.join(current_dir, "..", "models", "emotion2vec_plus_large")

# -----------------------------------------------------------------------------
# Speaker Verification
# -----------------------------------------------------------------------------

enroll_embeddings_json = os.path.join(current_dir, "..", "sample", "embeddings.json")
enroll_audio_dir = None
color_map = {"Unknown": "gray", "A": "green"}

# -----------------------------------------------------------------------------
# Spoof Detection
# -----------------------------------------------------------------------------

spoof_model_root = os.path.join(current_dir, "..", "models", "spoof_detection")

# -----------------------------------------------------------------------------
# Automatic Speech Recognition
# -----------------------------------------------------------------------------

asr_model_root = os.path.join(current_dir, "..", "models", "SenseVoiceSmall")

# -----------------------------------------------------------------------------
# Audio Source Separation
# -----------------------------------------------------------------------------

ckpt = os.path.join(
    current_dir, "..", "models", "zeroshot_separation", "zeroshot_asp_full.ckpt"
)
resume_ckpt = os.path.join(
    current_dir, "..", "models", "zeroshot_separation", "htsat_audioset_2048d.ckpt"
)
query_folder = os.path.join(current_dir, "..", "sample", "query")
query_sr = 16000
output_dir = os.path.join(current_dir, "..", "sample", "zsass_output")
output_filename = "pred.wav"
