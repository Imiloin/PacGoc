import numpy as np


def pcm16to32(audio: np.ndarray[np.int16]) -> np.ndarray[np.float32]:
    """
    Convert PCM 16-bit audio to 32-bit float audio.
    """
    assert audio.dtype == np.int16
    audio = audio.astype(np.float32)
    bits = np.iinfo(np.int16).bits
    audio = audio / (2 ** (bits - 1))
    return audio


def pcm32to16(audio: np.ndarray[np.float32]) -> np.ndarray[np.int16]:
    """
    Convert PCM 32-bit float audio to 16-bit audio.
    """
    assert audio.dtype == np.float32
    bits = np.iinfo(np.int16).bits
    audio = audio * (2 ** (bits - 1))
    audio = np.round(audio).astype(np.int16)
    return audio
