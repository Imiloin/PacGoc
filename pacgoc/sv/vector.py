import os
import json
from typing import Union
import librosa
import numpy as np
import torch
from modelscope.pipelines import pipeline
from ..utils import pcm16to32


class Vector:
    MODEL_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: int = 16000,
        isint16: bool = True,
        model_root: os.PathLike = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
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
        self.model_root = model_root
        self.threshold = threshold

        self.pipeline = pipeline(
            task="speaker-verification",
            model=model_root,
        )

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
                print("generating embedding for %s" % enroll_audio)
                audio = self.pipeline.preprocess([enroll_audio])
                enroll_embedding = self.pipeline.forward(audio)
                enroll_embedding = enroll_embedding.numpy()
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

    def _is_valid_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        if not os.path.isfile(json_file):
            return False
        if not json_file.endswith(".json"):
            return False
        return True

    def preprocess(self, audio_data: np.ndarray):
        """
        Preprocess the audio data, convert to float32 and resample to 16000Hz if necessary.
        """
        if self.isint16:
            audio_data = audio_data.view(dtype=np.int16)
            # convert to float32
            audio_data = pcm16to32(audio_data)
        if self.sr != Vector.MODEL_SAMPLE_RATE:
            # resample to 16000Hz
            audio_data = librosa.resample(
                audio_data,
                orig_sr=self.sr,
                target_sr=Vector.MODEL_SAMPLE_RATE,
                scale=True,
            )
        # audio_data = torch.from_numpy(audio_data) ######
        return audio_data

    def infer(self, data) -> torch.Tensor:
        """
        Forward the model to get the embedding.
        """
        return self.pipeline.model(data)

    def postprocess(
        self, input: torch.Tensor, file_name: str = "emb", save_dir: os.PathLike = None
    ):
        """
        Save the embedding to a npy file.
        """
        if save_dir is not None:
            # save the embeddings
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "%s.npy" % file_name)
            np.save(save_path, input.numpy())

    def compute_cos_similarity(
        self,
        emb1: Union[np.ndarray, torch.Tensor],
        emb2: Union[np.ndarray, torch.Tensor],
    ) -> float:
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        assert len(emb1.shape) == 2 and len(emb2.shape) == 2
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine = cos(emb1, emb2)
        return cosine.item()

    def verify(
        self, embedding: Union[np.ndarray, torch.Tensor], threshold: float = 0.7
    ) -> str:
        """
        Verify the speaker by the audio embedding.
        Return the speaker id if the score is above the threshold.
        """
        scores = []
        for enroll_audio, enroll_embedding in self.enroll_embeddings.items():
            enroll_embedding = np.array(enroll_embedding)
            enroll_embedding = torch.from_numpy(enroll_embedding)
            score = self.compute_cos_similarity(enroll_embedding, embedding)
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
        data = self.preprocess(audio_data)
        emmbedding = self.infer(data)
        speaker_id = self.verify(emmbedding)
        return speaker_id
