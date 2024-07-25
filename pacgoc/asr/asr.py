import os
import librosa
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
from typing import Any
from ..utils import pcm16to32


class ASR:
    MODEL_SAMPLE_RATE = 16000
    CHUNK_SIZE = 512

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = "iic/SenseVoiceSmall",
        lang: str = "auto",
    ):
        """
        Load the ASR model and initialize the language.
        """
        self.sr = sr
        self.isint16 = isint16

        if not os.path.exists(model_root):
            print(f"Model root {model_root} does not exist.")
            exit(1)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel(model=model_root, device=device)

        self.lang = lang  # "zn", "en", "yue", "ja", "ko", "nospeech"
        self.clear_cache()

    def clear_cache(self):
        self.cache = {}
        self.prev_chunk = np.empty(ASR.CHUNK_SIZE)

    def preprocess(self, audio_data: np.ndarray):
        """
        Preprocess the audio data by converting it to 16000 Hz, 32-bit float.
        """
        if self.isint16:
            audio = audio_data.view(dtype=np.int16)
            audio = pcm16to32(audio)
        else:
            audio = audio_data
        if self.sr != ASR.MODEL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=ASR.MODEL_SAMPLE_RATE, scale=True
            )
        # add previous chunk to the current audio
        audio = np.concatenate([self.prev_chunk, audio])
        return audio

    def infer(self, audio: np.ndarray) -> list | Any:
        """
        Perform inference on the given audio data.
        """
        res = self.model.generate(
            input=audio,
            cache=self.cache,
            language=self.lang,
            use_itn=True,
            # batch_size=64,
        )
        self.prev_chunk = audio[-ASR.CHUNK_SIZE:]
        return res

    def postprocess(self, res: list | Any) -> str:
        """
        Extract the text from the decoding result.
        """
        text = rich_transcription_postprocess(res[0]["text"])
        text = text[:-1]  # remove the last two characters "。<emo>"
        return text

    def __call__(self, audio_data: np.ndarray) -> str:
        """
        Perform ASR on the given audio data.
        """
        audio = self.preprocess(audio_data)
        infer_result = self.infer(audio)
        return self.postprocess(infer_result)
