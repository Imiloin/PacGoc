import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Recordings
# -----------------------------------------------------------------------------

recordings_dir = os.path.join(current_dir, "..", "recordings")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

share = False
username = "admin"
password = "admin"

INTERVAL = 10  # running interval, in seconds
MAX_AUDIO_LEN = 30  # maximum audio length, in seconds

# -----------------------------------------------------------------------------
# ON/OFF
# -----------------------------------------------------------------------------

HARDWARE_CONTROLLER_ON = True
AUDIO_CLASSIFICATION_ON = True
SPEAKER_PROFILING_ON = True
SPEAKER_VERIFICATION_ON = True
SPOOF_DETECTION_ON = True
AUTOMATIC_SPEECH_RECOGNITION_ON = True
ACOUSTIC_NOISE_SUPPRESSION_ON = True
AUDIO_SOURCE_SEPARATION_ON = True
LLM_CHAT_ON = True

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

# record & playback
RECORD_START = "RS"
RECORD_END = "RE"
PLAYBACK_START = "PS"
PLAYBACK_END = "PE"

# -----------------------------------------------------------------------------
# Audio Classification
# -----------------------------------------------------------------------------

cls_model_root = os.path.join(current_dir, "..", "models", "ced-base")

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

sv_model_root = os.path.join(
    current_dir, "..", "models", "speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
)
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
# Acoustic Noise Suppression
# -----------------------------------------------------------------------------

ans_model_root = os.path.join(current_dir, "..", "models", "speech_frcrn_ans_cirm_16k")
ans_output_dir = os.path.join(current_dir, "..", "sample", "ans_output")
ans_output_filename = "denoised.wav"

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
separation_output_dir = os.path.join(current_dir, "..", "sample", "zsass_output")
separation_output_filename = "pred.wav"

# -----------------------------------------------------------------------------
# LLM Chat
# -----------------------------------------------------------------------------

llm_model_root = os.path.join(current_dir, "..", "models", "Qwen2-1.5B-Instruct")
