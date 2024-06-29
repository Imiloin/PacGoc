from funasr import AutoModel
import os
import numpy as np
from ...utils import pcm16to32


################# to do: change the sampling rate to 16000
################# conflicting with paddle.utils.run_check() ????
class Emotion:
    def __init__(self, model: os.PathLike = "iic/emotion2vec_base_finetuned"):
        """
        Initialize the Emotion class.
        """
        self.model = AutoModel(model=model)

    def preprocess(self, audio_data: np.ndarray):
        audio = audio_data.view(dtype=np.int16)
        audio = pcm16to32(audio)
        # print(audio)
        # print(type(audio))
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
