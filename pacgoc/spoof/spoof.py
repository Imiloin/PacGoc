import os
import numpy as np
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from ..utils import pcm16to32


class SpoofDetector:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = None,
    ):
        """
        Init model and other resources.
        """
        self.sr = sr
        self.isint16 = isint16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.id2label = {"0": "bonafide", "1": "spoof"}

        # get feature extractor from pre-trained model
        model_id = os.path.join(model_root, "pre-trained")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, do_normalize=True, return_attention_mask=True
        )

        # load fine-tuned model
        model_path = os.path.join(model_root, "fine-tuned")
        self.model = AutoModelForAudioClassification.from_pretrained(model_path).to(
            self.device
        )

    def preprocess(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess the audio data, convert to float32 and resample to 16000Hz if necessary.
        """
        if self.isint16:
            audio_data = audio_data.view(dtype=np.int16)
            # convert to float32
            audio_data = pcm16to32(audio_data)
        if self.sr != SpoofDetector.MODEL_SAMPLE_RATE:
            # resample to 16000Hz
            audio_data = librosa.resample(
                audio_data,
                orig_sr=self.sr,
                target_sr=SpoofDetector.MODEL_SAMPLE_RATE,
                scale=True,
            )
        # extract features
        inputs = self.feature_extractor(
            audio_data, sampling_rate=16000, return_tensors="pt", padding=True
        )
        # convert to tensor and move to device
        input_values = inputs.input_values.to(self.device)
        return input_values

    def infer(self, input_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(input_values).logits
        return logits

    def postprocess(self, logits: torch.Tensor) -> str:
        """
        Postprocess the model output to get the predicted label.
        """
        # print(torch.sigmoid(logits))
        predicted_ids = torch.argmax(logits, dim=-1)
        res = self.id2label[str(predicted_ids.item())]
        # "bonafide" or "spoof"
        return res

    def __call__(self, audio_data: np.ndarray) -> str:
        """
        Input preprocess, model inference and output postprocess.
        """
        input_values = self.preprocess(audio_data)
        logits = self.infer(input_values)
        res = self.postprocess(logits)
        return res
