import os
import librosa
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from ..utils import pcm16to32

from modelscope.pipelines.audio import ANSPipeline
from .patch import custom_preprocess, custom_forward, custom_postprocess


class ANS:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = "damo/speech_frcrn_ans_cirm_16k",
        output_path: os.PathLike = None,
        output_filename: str = "denoised.wav",
    ):
        """
        Initialize the Emotion class.
        """
        self.sr = sr
        self.isint16 = isint16

        if not os.path.exists(model_root):
            print(f"Model root {model_root} does not exist.")
            exit(1)

        assert output_path is not None, "output_path should not be None"
        os.makedirs(output_path, exist_ok=True)
        self.output_file = os.path.join(output_path, output_filename)

        self._patch()

        self.pipeline = pipeline(Tasks.acoustic_noise_suppression, model=model_root)

    def _patch(self):
        """
        Monkey Patch the ANS pipeline to support np.ndarray input.
        """
        ANSPipeline.preprocess = custom_preprocess
        ANSPipeline.forward = custom_forward
        ANSPipeline.postprocess = custom_postprocess

    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess the audio data, convert it to 16kHz float32 np.ndarray.
        """
        if self.isint16:
            audio = audio_data.view(dtype=np.int16)
            audio = pcm16to32(audio)
        else:
            audio = audio_data
        if self.sr != ANS.MODEL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=ANS.MODEL_SAMPLE_RATE, scale=True
            )
        return audio

    def inference(self, audio: np.ndarray):
        """
        Run the inference on the audio data using the ANS pipeline.
        """
        result = self.pipeline(audio, output_path=self.output_file)
        return result
    
    def postprocess(self):
        pass
    
    def __call__(self, audio_data: np.ndarray) -> os.PathLike:
        """
        Run the entire pipeline on the audio data.
        """
        audio = self.preprocess(audio_data)
        self.inference(audio)
        self.postprocess()
        return self.output_file
