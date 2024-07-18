import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from .model import AgeGenderModel
from ...utils import pcm16to32


class AgeGender:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = None,
    ):
        """
        Initialize AgeGender model, if not exists, download and extract it.
        """
        self.sr = sr
        self.isint16 = isint16
        
        self.labels = {"age": 0, "female": 1, "male": 2, "child": 3}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_root):
            print(f"Model root {model_root} does not exist.")
            exit(1)

        self.processor = Wav2Vec2Processor.from_pretrained(model_root)
        self.model = AgeGenderModel.from_pretrained(model_root)
        self.model = self.model.to(self.device)

    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess the audio data, convert to float32 and resample to 16000Hz if necessary.
        """
        if self.isint16:
            audio_data = audio_data.view(dtype=np.int16)
            # convert to float32
            audio_data = pcm16to32(audio_data)
        if self.sr != AgeGender.MODEL_SAMPLE_RATE:
            # resample to 16000Hz
            audio_data = librosa.resample(
                audio_data,
                orig_sr=self.sr,
                target_sr=AgeGender.MODEL_SAMPLE_RATE,
                scale=True,
            )
        return audio_data

    def infer(
        self,
        audio_data: np.ndarray[np.float32],
        embeddings: bool = False,
    ) -> np.ndarray:
        """
        Create interface for feature extraction and infer age and gender
        """
        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(audio_data, sampling_rate=AgeGender.MODEL_SAMPLE_RATE)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y)
            if embeddings:
                y = y[0]
            else:
                y = torch.hstack([y[1], y[2]])

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y

    def postprocess(self, infer_result: np.ndarray) -> dict:
        """
        Postprocess the inference result, return a dictionary with age and gender.
        """
        res = {}
        infer_result = infer_result.flatten()
        # ignore child class
        if infer_result[self.labels["female"]] < infer_result[self.labels["male"]]:
            res["gender"] = "男/Male"
        else:
            res["gender"] = "女/Female"
        res["age"] = round(infer_result[self.labels["age"]] * 100)
        return res

    def __call__(self, audio_data: np.ndarray) -> dict:
        """
        Call the model to infer age and gender
        """
        preprocessed_data = self.preprocess(audio_data)
        infer_result = self.infer(preprocessed_data)
        return self.postprocess(infer_result)
