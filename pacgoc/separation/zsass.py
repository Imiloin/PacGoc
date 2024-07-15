import os
import sys
from pathlib import Path

# add current directory to sys.path to import pcie module
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import librosa
import numpy as np

import torch
from torch.utils.data import DataLoader
from utils import create_folder, prepprocess_audio
from models.asp_model import ZeroShotASP, AutoTaggingWarpper
from models.separator import SeparatorModel
from data_processor import MusdbDataset

import asp_config
import htsat_config
from models.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
import pytorch_lightning as pl

from ..utils import pcm16to32
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
        pl.utilities.seed.seed_everything(seed=12412)  # set the random seed
        self.test_key = ["vocals"]  # ["vocals", "drums", "bass", "other"]
        self.config = asp_config

        # setup the trainer and enabel GPU
        self.trainer = pl.Trainer(gpus=1, accelerator="auto")

        assert ckpt is not None, "there should be a saved model when inferring"
        self.ckpt = ckpt
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
        self.trainer.test(at_wrapper, test_dataloaders=avg_loader)
        self.avg_at = at_wrapper.avg_at

    def preprocess(self, audio_data: np.ndarray):
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
        return audio

    def inference(self, audio: np.ndarray):
        """
        Perform source separation on the given audio data.
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

        # import seapration model
        model = ZeroShotASP(
            channels=1, config=self.config, at_model=self.at_model, dataset=dataset
        )
        # resume checkpoint
        ckpt = torch.load(self.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        exp_model = SeparatorModel(
            model=model,
            config=self.config,
            target_keys=self.test_key,
            avg_at=self.avg_at,
            using_wiener=False,
            calc_sdr=False,
            output_wav=True,
        )
        self.trainer.test(exp_model, test_dataloaders=loader)

    def postprocess(self):
        pass

    def __call__(self, audio_data: np.ndarray):
        """
        The main function to perform source separation.
        Generate the separated an audio file and save it to the output_path.
        """
        audio = self.preprocess(audio_data)
        self.inference(audio)
        self.postprocess()
