import whisper
import numpy as np
from ..utils import pcm16to32
from typing import List, Union
from whisper.decoding import DecodingResult


class ASR:
    def __init__(self, model: str = "medium"):
        """
        Load the ASR model and initialize the language.
        """
        self.model = whisper.load_model(model)
        self.lang = "zh"
        self.prev_lang = "zh"
        self.options = whisper.DecodingOptions(without_timestamps=True)
        self.result = None

    def preprocess(self, audio_data: np.ndarray):
        """
        Preprocess the audio data by converting it to 32-bit float.
        """
        audio = audio_data.view(dtype=np.int16)
        audio = pcm16to32(audio)
        return audio

    def infer(self, audio: np.ndarray) -> Union[DecodingResult, List[DecodingResult]]:
        """
        Perform inference on the given audio data.
        """
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        self.lang = max(probs, key=probs.get)
        if (not self.result or self.prev_lang != "zh") and self.lang == "zh":
            self.options = whisper.DecodingOptions(
                prompt="我在说简体中文哦，", without_timestamps=True
            )
        self.prev_lang = self.lang

        # decode the audio
        self.result = whisper.decode(self.model, mel, self.options)
        # print(f"{self.lang}: {result.text}")
        return self.result

    def postprocess(self, result: DecodingResult) -> str:
        """
        Extract the text from the decoding result.
        """
        return result.text

    def __call__(self, audio_data: np.ndarray) -> str:
        """
        Perform ASR on the given audio data.
        """
        audio = self.preprocess(audio_data)
        infer_result = self.infer(audio)
        return self.postprocess(infer_result)
