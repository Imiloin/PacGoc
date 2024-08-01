import os
import wave
import numpy as np


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
