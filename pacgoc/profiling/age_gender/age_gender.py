import os
import audeer
import audonnx
import audinterface
import librosa
import numpy as np
from ...utils import pcm16to32


class AgeGender:
    MODEL_SAMPLE_RATE = 16000
    url = "https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"

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

        # if model is not specified, download and extract model
        if not os.path.exists(model_root):
            print("Downloading and extracting AgeGender model...")
            model_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "model"
            )
            cache_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cache"
            )
            audeer.mkdir(cache_root)
            dst_path = os.path.join(cache_root, "model.zip")
            audeer.download_url(AgeGender.url, dst_path, verbose=True)
            audeer.extract_archive(dst_path, model_root, verbose=True)

        self.model = audonnx.load(model_root)

    def preprocess(self, audio_data: np.ndarray):
        """
        Preprocess the audio data, convert to float32 and resample to 16000Hz if necessary.
        """
        if self.isint16:
            audio_data = audio_data.view(dtype=np.int16)
            # convert to float32
            audio_data = pcm16to32(audio_data)
        # if self.sr != AgeGender.MODEL_SAMPLE_RATE:
        #     audio_data = librosa.resample(
        #         audio_data, orig_sr=self.sr, target_sr=AgeGender.MODEL_SAMPLE_RATE, scale=True
        #     )
        return audio_data

    def infer(self, audio_data: np.ndarray[np.float32]):
        """
        Create interface for feature extraction and infer age and gender
        """
        outputs = ["logits_age", "logits_gender"]
        self.interface = audinterface.Feature(
            self.model.labels(outputs),
            process_func=self.model,
            process_func_args={
                "outputs": outputs,
                "concat": True,
            },
            sampling_rate=self.sr,
            resample=True,  # auto resample
            verbose=True,
        )
        return self.interface.process_signal(audio_data, self.sr)

    def postprocess(self, infer_result) -> dict:
        """
        Postprocess the inference result, return a dictionary with age and gender.
        """
        res = {}
        # ignore child class
        if infer_result["female"].iloc[0] < infer_result["male"].iloc[0]:
            res["gender"] = "男/Male"
        else:
            res["gender"] = "女/Female"
        res["age"] = round(infer_result["age"].iloc[0] * 100)
        return res

    def __call__(self, audio_data: np.ndarray) -> dict:
        """
        Call the model to infer age and gender
        """
        preprocessed_data = self.preprocess(audio_data)
        infer_result = self.infer(preprocessed_data)
        return self.postprocess(infer_result)
