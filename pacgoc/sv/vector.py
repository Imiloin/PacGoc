import os
import json
from typing import Union
import paddle
import librosa
import numpy as np
from yacs.config import CfgNode
from paddlespeech.cli.vector.infer import VectorExecutor
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddleaudio.compliance.librosa import melspectrogram
from paddlespeech.vector.io.batch import feature_normalize
from ..utils import pcm16to32


class Vector:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        threshold: float = 0.7,
        enroll_embeddings: os.PathLike = None,
        enroll_audio_dir: os.PathLike = None,
    ):
        """
        Initialize enroll embeddings.
        enroll_embeddings and enroll_embeddings can not be both None.
        If enroll_audio_dir is not None, generate embeddings for all enroll audios in the directory.
        If enroll_embeddings is not None, load embeddings from the json file.
        If enroll_embeddings is not None and enroll_audio_dir is not None, save embeddings to the json file.
        """
        self.sr = sr
        self.isint16 = isint16
        self.threshold = threshold

        self.executor = VectorExecutor()
        device = paddle.get_device()
        paddle.set_device(device)
        if device == "cpu":
            print("Vector: device is cpu")

        if enroll_embeddings is None and enroll_audio_dir is None:
            raise ValueError("Please provide enroll_embeddings or enroll_audio_dir.")
        elif enroll_audio_dir is not None:
            if not os.path.exists(enroll_audio_dir):
                raise ValueError("enroll_audio_dir not exists.")
            wav_files = [
                os.path.join(enroll_audio_dir, file)
                for file in os.listdir(enroll_audio_dir)
                if file.endswith(".wav")
            ]
            if len(wav_files) == 0:
                print("No wav files found in enroll_audio_dir.")

            # Generate embeddings for all enroll audios and save them to a json file
            self.enroll_embeddings = {}
            for enroll_audio in wav_files:
                enroll_embedding = self.executor(enroll_audio)
                audio_file_name_with_extension = os.path.basename(enroll_audio)
                audio_file_name, _ = os.path.splitext(audio_file_name_with_extension)
                self.enroll_embeddings[audio_file_name] = (
                    enroll_embedding.tolist()
                )  # convert numpy array to list
            if enroll_embeddings is not None:
                if enroll_embeddings.endswith(".json"):
                    with open(enroll_embeddings, "w") as f:
                        json.dump(self.enroll_embeddings, f)
                else:
                    print("Invalid enroll_embeddings file path.")
        else:
            if self._is_valid_json(enroll_embeddings):
                with open(enroll_embeddings, "r") as f:
                    self.enroll_embeddings = json.load(f)
            else:
                raise ValueError("Invalid enroll_embeddings file path.")

        self._init_executor()

    def _is_valid_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        if not os.path.isfile(json_file):
            return False
        if not json_file.endswith(".json"):
            return False
        return True

    def _init_executor(
        self,
        model_type: str = "ecapatdnn_voxceleb12",
        sample_rate: int = 16000,
        task=None,
    ):
        """
        Init the neural network model and load the model parameters.
        """
        # stage 0: avoid to init the mode again
        self.executor.task = task
        if hasattr(self.executor, "model"):
            return

        # stage 1: get the model and config path
        #          if we want init the network from the model stored in the disk,
        #          we must pass the config path and the ckpt model path
        # get the mode from pretrained list
        sample_rate_str = "16k" if sample_rate == 16000 else "8k"
        tag = model_type + "-" + sample_rate_str
        self.executor.task_resource.set_task_model(tag, version=None)
        # get the model from the pretrained list
        # we download the pretrained model and store it in the res_path
        self.executor.res_path = self.executor.task_resource.res_dir

        self.executor.cfg_path = os.path.join(
            self.executor.task_resource.res_dir,
            self.executor.task_resource.res_dict["cfg_path"],
        )
        self.executor.ckpt_path = os.path.join(
            self.executor.task_resource.res_dir,
            self.executor.task_resource.res_dict["ckpt_path"] + ".pdparams",
        )

        # stage 2: read and config and init the model body
        self.executor.config = CfgNode(new_allowed=True)
        self.executor.config.merge_from_file(self.executor.cfg_path)

        # stage 3: get the model name to instance the model network with dynamic_import
        model_name = model_type[: model_type.rindex("_")]
        model_class = self.executor.task_resource.get_model_class(model_name)
        model_conf = self.executor.config.model
        backbone = model_class(**model_conf)
        model = SpeakerIdetification(
            backbone=backbone, num_class=self.executor.config.num_speakers
        )
        self.executor.model = model
        self.executor.model.eval()

        # stage 4: load the model parameters
        model_dict = paddle.load(self.executor.ckpt_path)
        self.executor.model.set_state_dict(model_dict)

    def preprocess(self, audio_data: np.ndarray):
        """Extract the audio feature from the audio data."""
        # stage 1: load the audio sample points
        if self.isint16:
            waveform = audio_data.view(dtype=np.int16)
            # convert to float32
            waveform = pcm16to32(waveform)
        else:
            waveform = audio_data

        if self.sr != Vector.MODEL_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=self.sr, target_sr=Vector.MODEL_SAMPLE_RATE, scale=True
            )

        # stage 2: get the audio feat
        # Note: Now we only support fbank feature
        try:
            feat = melspectrogram(
                x=waveform,
                sr=self.executor.config.sr,
                n_mels=self.executor.config.n_mels,
                window_size=self.executor.config.window_size,
                hop_length=self.executor.config.hop_size,
            )
        except Exception as e:
            print(f"feat occurs exception {e}")

        feat = paddle.to_tensor(feat).unsqueeze(0)
        # in inference period, the lengths is all one without padding
        lengths = paddle.ones([1])

        # stage 3: we do feature normalize,
        #          Now we assume that the feat must do normalize
        feat = feature_normalize(feat, mean_norm=True, std_norm=False)

        # stage 4: store the feat and length in the _inputs,
        #          which will be used in other function
        self.executor._inputs["feats"] = feat
        self.executor._inputs["lengths"] = lengths

    def infer(self):
        """Infer the model to get the embedding

        Args:
            model_type (str): speaker verification model type
        """
        # stage 0: get the feat and length from _inputs
        feats = self.executor._inputs["feats"]
        lengths = self.executor._inputs["lengths"]

        # stage 1: get the audio embedding
        # embedding from (1, emb_size, 1) -> (emb_size)
        embedding = self.executor.model.backbone(feats, lengths).squeeze().numpy()

        # stage 2: put the embedding and dim info to _outputs property
        #          the embedding type is numpy.array
        self.executor._outputs["embedding"] = embedding

    def postprocess(self) -> Union[str, os.PathLike]:
        """
        Return the audio embedding info.
        """
        embedding = self.executor._outputs["embedding"]
        return embedding

    def verify(self, embedding: Union[str, os.PathLike], threshold: float = 0.7) -> str:
        """
        Verify the speaker by the audio embedding.
        Return the speaker id if the score is above the threshold.
        """
        scores = []
        for enroll_audio, enroll_embedding in self.enroll_embeddings.items():
            score = self.executor.get_embeddings_score(enroll_embedding, embedding)
            scores.append((enroll_audio, score))

        ## Sort the scores in descending order and get the top 3
        # top_3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

        # for audio, score in top_3:
        #     print(f"Score of {audio} is {score}")

        best_match = sorted(scores, key=lambda x: x[1], reverse=True)[0]
        speaker_id = os.path.basename(best_match[0]).split(".")[0]
        best_score = best_match[1]

        # Check if the score is above a certain threshold
        if best_score < self.threshold:
            return "Unknown"
        else:
            return speaker_id

    def __call__(self, audio_data: np.ndarray) -> str:
        self.preprocess(audio_data)
        self.infer()
        emmbedding = self.postprocess()
        speaker = self.verify(emmbedding)
        return speaker
