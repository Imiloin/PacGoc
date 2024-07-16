import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Recordings
# -----------------------------------------------------------------------------

recordings_dir = os.path.join(current_dir, "..", "recordings")

# -----------------------------------------------------------------------------
# ON/OFF
# -----------------------------------------------------------------------------

AUDIO_CLASSIFIER_ON = True
SPEAKER_PROFILING_ON = True
SPEAKER_VERIFICATION_ON = True
SPOOF_DETECTION_ON = True
AUTOMATIC_SPEECH_RECOGNITION_ON = True
AUDIO_SOURCE_SEPARATION_ON = True

# -----------------------------------------------------------------------------
# Speaker Profiling
# -----------------------------------------------------------------------------

age_gender_model_root = os.path.join(current_dir, "..", "models", "age_gender")

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

asr_model_type = "medium"

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
