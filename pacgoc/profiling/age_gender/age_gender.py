import os
import audeer
import audonnx
import audinterface
import numpy as np


################# to do: change the sampling rate to 16000
class AgeGender:
    url = "https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"

    def __init__(
        self,
        sampling_rate: int = 16000,
        model_root: os.PathLike = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "model"
        ),
        cache_root: os.PathLike = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"
        ),
    ):
        """
        Initialize AgeGender model, if not exists, download and extract it.
        """
        self.sampling_rate = sampling_rate
        # download and extract model
        audeer.mkdir(cache_root)
        dst_path = os.path.join(cache_root, "model.zip")

        if not os.path.exists(dst_path):
            audeer.download_url(AgeGender.url, dst_path, verbose=True)

        if not os.path.exists(model_root):
            audeer.extract_archive(dst_path, model_root, verbose=True)

        self.model = audonnx.load(model_root)

    def preprocess(self, audio_data: np.ndarray):
        """
        Create interface for feature extraction
        """
        outputs = ["logits_age", "logits_gender"]
        self.interface = audinterface.Feature(
            self.model.labels(outputs),
            process_func=self.model,
            process_func_args={
                "outputs": outputs,
                "concat": True,
            },
            sampling_rate=self.sampling_rate,
            resample=True,
            verbose=True,
        )
        audio_data = audio_data.view(dtype=np.int16)
        audio_data = audio_data.astype(np.float32)
        return audio_data

    def infer(self, audio_data: np.ndarray[np.float32]):
        """
        Infer age and gender
        """
        return self.interface.process_signal(audio_data, self.sampling_rate)

    def postprocess(self, infer_result) -> dict:
        """
        Postprocess the inference result, return a dictionary with age and gender.
        """
        res = {}
        if infer_result["female"].iloc[0] < infer_result["male"].iloc[0]:
            res["sex"] = "male"
        else:
            res["sex"] = "female"
        res["age"] = round(infer_result["age"].iloc[0] * 100)
        return res

    def __call__(self, audio_data: np.ndarray) -> dict:
        """
        Call the model to infer age and gender
        """
        preprocessed_data = self.preprocess(audio_data)
        infer_result = self.infer(preprocessed_data)
        return self.postprocess(infer_result)
