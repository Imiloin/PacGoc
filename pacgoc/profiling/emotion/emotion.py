from funasr import AutoModel
import os
import librosa
import numpy as np
from ...utils import pcm16to32


################# conflicting with paddle.utils.run_check() ????
class Emotion:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model: os.PathLike = "iic/emotion2vec_plus_base",
    ):
        """
        Initialize the Emotion class.
        """
        self.sr = sr
        self.isint16 = isint16
        self.model = AutoModel(model=model)

    def preprocess(self, audio_data: np.ndarray):
        if self.isint16:
            audio = audio_data.view(dtype=np.int16)
            audio = pcm16to32(audio)
        else:
            audio = audio_data
        if self.sr != Emotion.MODEL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=Emotion.MODEL_SAMPLE_RATE
            )
        return audio

    def infer(self, audio: np.ndarray[np.float32], output_dir: os.PathLike = None):
        """
        Use the emotion2vec model to infer the emotion of the input audio data
        """
        rec_result = self.model.generate(
            audio, output_dir=output_dir, granularity="utterance"
        )
        return rec_result

    def postprocess(self, rec_result):
        """
        Return the emotion with the highest score
        """
        idx = rec_result[0]["scores"].index(max(rec_result[0]["scores"]))
        return rec_result[0]["labels"][idx]

    def __call__(self, audio_data: np.ndarray) -> str:
        """
        Call the model to infer speech emotion
        """
        preprocessed_data = self.preprocess(audio_data)
        infer_result = self.infer(preprocessed_data)
        return self.postprocess(infer_result)
