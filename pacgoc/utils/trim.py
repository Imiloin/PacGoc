import numpy as np


def trim_start_silence(audio_data: np.ndarray[np.float32], threshold=1e-4):
    """
    Trim the start of the audio data to remove any silence at the beginning.

    Args:
        audio_data (np.ndarray[np.float32]): The audio data to trim.
        threshold (float, optional): The threshold for detecting silence. Defaults to 1e-4.

    Returns:
        np.ndarray[np.float32]: The trimmed audio data.
    """
    # calculate the energy of the audio signal
    energy = np.square(audio_data)
    # find the index of the first non-silence frame
    start_index = np.argmax(energy > threshold)
    # trim the audio data
    trimmed_audio = audio_data[start_index:]

    return trimmed_audio
