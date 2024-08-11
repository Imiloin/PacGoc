import os
import librosa
import numpy as np

import torch
from torch.utils.data import DataLoader
from .utils import create_folder, prepprocess_audio
from .models.asp_model import ZeroShotASP, AutoTaggingWarpper
from .models.separator import SeparatorModel
from .data_processor import MusdbDataset

from . import asp_config
from . import htsat_config
from .models.htsat import HTSAT_Swin_Transformer
from .sed_model import SEDWrapper
import pytorch_lightning as pl

from ..utils import pcm16to32
from ..utils import trim_start_silence
import warnings

warnings.filterwarnings("ignore")


class SourceSeparation:
    MODEL_SAMPLE_RATE = 16000  ####### or 32000 Hz?

    def __init__(
        self,
        sr: int = 16000,
        query_sr: int = 16000,
        isint16: bool = False,
        ckpt: os.PathLike = None,
        resume_ckpt: os.PathLike = None,
        query_folder: os.PathLike = None,
        output_path: os.PathLike = None,
        output_filename: str = "pred.wav",
    ):
        """
        Initialize the source separation model, setups the config and load the saved model.
        """
        self.sr = sr
        self.query_sr = query_sr
        self.isint16 = isint16
        self.test_key = ["vocals"]  # ["vocals", "drums", "bass", "other"]
        self.config = asp_config

        # setup the trainer and enabel GPU
        accelerator = "gpu" if torch.cuda.is_available() else "auto"
        self.trainer = pl.Trainer(accelerator=accelerator, devices="auto")

        assert ckpt is not None, "there should be a saved model when inferring"
        self.ckpt = torch.load(ckpt, map_location="cpu")
        htsat_config.resume_checkpoint = resume_ckpt

        # obtain the samples for query
        queries = []
        for query_file in os.listdir(query_folder):
            f_path = os.path.join(query_folder, query_file)
            if query_file.endswith(".wav"):
                temp_q, fs = librosa.load(f_path, sr=None)
                temp_q = temp_q[:, None]
                temp_q = prepprocess_audio(temp_q, fs, self.query_sr, "mix")
                temp = [temp_q]
                for _ in self.test_key:
                    temp.append(temp_q)
                temp = np.array(temp)
                queries.append(temp)
        assert (
            len(queries) > 0
        ), "There should be at least one query audio file in the inference_query folder"

        assert output_path is not None, "output_path should not be None"
        self.config.wave_output_path = output_path
        self.config.output_filename = output_filename

        create_folder(output_path)

        sed_model = HTSAT_Swin_Transformer(
            spec_size=htsat_config.htsat_spec_size,
            patch_size=htsat_config.htsat_patch_size,
            in_chans=1,
            num_classes=htsat_config.classes_num,
            window_size=htsat_config.htsat_window_size,
            config=htsat_config,
            depths=htsat_config.htsat_depth,
            embed_dim=htsat_config.htsat_dim,
            patch_stride=htsat_config.htsat_stride,
            num_heads=htsat_config.htsat_num_head,
        )
        self.at_model = SEDWrapper(
            sed_model=sed_model, config=htsat_config, dataset=None
        )
        htsat_ckpt = torch.load(htsat_config.resume_checkpoint, map_location="cpu")
        self.at_model.load_state_dict(htsat_ckpt["state_dict"])

        # obtain the latent embedding as query
        self.avg_at = None
        avg_dataset = MusdbDataset(tracks=queries)
        avg_loader = DataLoader(
            dataset=avg_dataset, num_workers=1, batch_size=1, shuffle=False
        )
        at_wrapper = AutoTaggingWarpper(
            at_model=self.at_model, config=self.config, target_keys=self.test_key
        )
        self.trainer.test(at_wrapper, dataloaders=avg_loader)
        self.avg_at = at_wrapper.avg_at
        
        # pre-load the model
        empty_dataset = MusdbDataset(
            tracks=[]
        )
        self.model = ZeroShotASP(
            channels=1, config=self.config, at_model=self.at_model, dataset=empty_dataset
        )
        self.model.load_state_dict(self.ckpt["state_dict"], strict=False)
        self.exp_model = SeparatorModel(
            model=self.model,
            config=self.config,
            target_keys=self.test_key,
            avg_at=self.avg_at,
            using_wiener=False,
            calc_sdr=False,
            output_wav=True,
        )

    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess the audio data by converting it to 32-bit float.
        """
        if self.isint16:
            audio = audio_data.view(dtype=np.int16)
            audio = pcm16to32(audio)
        else:
            audio = audio_data
        if self.sr != SourceSeparation.MODEL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=SourceSeparation.MODEL_SAMPLE_RATE, scale=True
            )
        # trim the start silence
        audio = trim_start_silence(audio, threshold=1e-4)  ####### threshold?
        return audio

    def inference(self, audio: np.ndarray):
        """
        Perform source separation on the given audio data.
        Will generate the separated audio files and save them to the output_path.
        """
        test_tracks = []
        temp = [audio]
        for _ in self.test_key:
            temp.append(audio)
        temp = np.array(temp)
        test_tracks.append(temp)
        dataset = MusdbDataset(
            tracks=test_tracks
        )  # the action is similar to musdbdataset, reuse it
        loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

        # set the dataset for the model and run the inference
        self.model.dataset = dataset
        self.exp_model.model = self.model
        self.trainer.test(self.exp_model, dataloaders=loader)

    def postprocess(self):
        pass
    
    def __str__(self):
        return os.path.join(self.output_path, self.output_filename)

    def __call__(self, audio_data: np.ndarray):
        """
        The main function to perform source separation.
        Generate the separated an audio file and save it to the output_path.
        """
        audio = self.preprocess(audio_data)
        self.inference(audio)
        self.postprocess()
