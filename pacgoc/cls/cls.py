import os
import numpy as np
import librosa
import torch
from .ced_model.feature_extraction_ced import CedFeatureExtractor
from .ced_model.modeling_ced import CedForAudioClassification
from ..utils import pcm16to32


class CLS:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = "mispeech/ced-base",
        topk: int = 3,
    ):
        """
        Init model and feature extractor.
        """
        self.sr = sr
        self.isint16 = isint16
        self.topk = topk

        if not os.path.exists(model_root) and model_root != "mispeech/ced-base":
            print(f"Model root {model_root} does not exist.")
            exit(1)
        self.feature_extractor = CedFeatureExtractor.from_pretrained(model_root)
        self.model = CedForAudioClassification.from_pretrained(model_root)

    def preprocess(self, audio_data: np.ndarray):
        """
        Preprocess the audio data, convert to float32 and resample to 16000Hz if necessary.
        """
        if self.isint16:
            audio_data = audio_data.view(dtype=np.int16)
            # convert to float32
            audio_data = pcm16to32(audio_data)
        if self.sr != CLS.MODEL_SAMPLE_RATE:
            # resample to 16000Hz
            audio_data = librosa.resample(
                audio_data, orig_sr=self.sr, target_sr=CLS.MODEL_SAMPLE_RATE, scale=True
            )
        return audio_data

    def infer(self, aduio_data: np.ndarray):
        """
        Model inference
        """
        inputs = self.feature_extractor(
            aduio_data, sampling_rate=CLS.MODEL_SAMPLE_RATE, return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def postprocess(self, logits: torch.Tensor) -> list[tuple]:
        """
        Output postprocess and return human-readable results.
        Return a list of tuples, each tuple contains a label and a score.
        """
        top_k_logits, top_k_indices = torch.topk(logits, self.topk, dim=-1)

        # get top k labels and scores and format to list of tuples
        res = []
        for i in range(self.topk):
            class_id = top_k_indices[0, i].item()
            score = top_k_logits[0, i].item()
            label = self.model.config.id2label[class_id]
            res.append((label, score))

        # [('Speech', 0.8502144), ('Speech synthesizer', 0.27297658), ('Narration, monologue', 0.117594205)]
        return res

    def __call__(self, audio_data: np.ndarray) -> list:
        """
        Input preprocess, model inference and output postprocess.
        """
        audio_data = self.preprocess(audio_data)
        logits = self.infer(audio_data)
        return self.postprocess(logits)
