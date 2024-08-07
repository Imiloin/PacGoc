from funasr import AutoModel
import os
import librosa
import numpy as np
from ...utils import pcm16to32


class Emotion:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = "iic/emotion2vec_plus_large",
    ):
        """
        Initialize the Emotion class.
        """
        self.sr = sr
        self.isint16 = isint16

        if not os.path.exists(model_root):
            print(f"Model root {model_root} does not exist.")
            exit(1)

        self.model = AutoModel(model=model_root)

        # run a test
        if isint16:
            test_audio = np.zeros(shape=(Emotion.MODEL_SAMPLE_RATE,), dtype=np.int16)
        else:
            test_audio = np.zeros(shape=(Emotion.MODEL_SAMPLE_RATE,), dtype=np.float32)
        self(test_audio)
        print("Emotion recognition model loaded")

    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess the audio data, convert it to 16kHz float32 np.ndarray.
        """
        if self.isint16:
            audio = audio_data.view(dtype=np.int16)
            audio = pcm16to32(audio)
        else:
            audio = audio_data
        if self.sr != Emotion.MODEL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=Emotion.MODEL_SAMPLE_RATE, scale=True
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

    def postprocess(self, rec_result) -> str:
        """
        Return the emotion with the highest score
        生气/angry
        厌恶/disgusted
        恐惧/fearful
        开心/happy
        中立/neutral
        其他/other
        难过/sad
        吃惊/surprised
        <unk>
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
