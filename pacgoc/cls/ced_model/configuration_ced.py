# coding=utf-8
# Copyright 2023 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" CED model configuration"""


import os
from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.utils.hub import cached_file

logger = logging.get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

CED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mispeech/ced-tiny": os.path.join(current_dir, "config.json"),
}


class CedConfig(PretrainedConfig):
    model_type = "ced"

    r"""
    Configuration class for the CED model.

    Args:
        name (str, optional, *optional*):
            Name of the pre-defined configuration. Can be "ced-tiny", "ced-mini", "ced-small" or "ced-base".
        attn_drop_rate (float, *optional*, defaults to 0.0):
            Dropout probability for attention weights. Default to 0.0.
        depth (int, *optional*, defaults to 12): Number of transformer layers. Default to 12.
        drop_path_rate (float, *optional*, defaults to 0.0): Drop path is taken from timm. Default to 0.0.
        drop_rate (float, *optional*, defaults to 0.0):
            Dropout probability for input embeddings. Default to 0.0.
        embed_dim (int, *optional*, defaults to 768):
            Dimensionality of the audio patch embeddings. Default to 768.
        eval_avg (str, *optional*, defaults to `"mean"`):
            Type of pooling to use for evaluation. Can be "mean", "token", "dm" or "logit". Default to "mean".
        mlp_ratio (float, *optional*, defaults to 4.0):
            Ratio of hidden size in the feedforward layer to the embedding size. Default to 4.0.
        num_heads (int, *optional*, defaults to 12): Number of attention heads. Default to 12.
        outputdim (int, *optional*, defaults to 527): Dimensionality of the output. Default to 527.
        patch_size (int, *optional*, defaults to 16): Size of the patches. Default to 16.
        patch_stride (int, *optional*, defaults to 16): Stride of the patches. Default to 16.
        pooling (str, *optional*, defaults to `"mean"`):
            Type of pooling to use for the output. Can be "mean", "token", "dm" or "logit". Default to "mean".
        qkv_bias (bool, *optional*, defaults to `True`):
            Whether to include bias terms in the query, key and value projections. Default to True.
        target_length (int, *optional*, defaults to 1012): Frames of an audio chunk. Default to 1012.
    """

    def __init__(
        self,
        name=None,
        attn_drop_rate=0.0,
        depth=12,
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dim=768,
        eval_avg="mean",
        mlp_ratio=4.0,
        num_heads=12,
        outputdim=527,
        patch_size=16,
        patch_stride=16,
        pooling="mean",
        qkv_bias=True,
        target_length=1012,
        **kwargs,
    ):
        r"""
        TODO: Add docstring
        """

        super().__init__(**kwargs)

        if name == "ced-tiny":
            embed_dim = 192
            num_heads = 3
        elif name == "ced-mini":
            embed_dim = 256
            num_heads = 4
        elif name == "ced-small":
            embed_dim = 384
            num_heads = 6
        elif name == "ced-base":
            embed_dim = 768
            num_heads = 12
        else:
            logger.info("No model name specified for CedConfig, use default settings.")

        assert pooling in ("mean", "token", "dm", "logit")
        self.name = name
        self.attn_drop_rate = attn_drop_rate
        self.center = kwargs.get("center", True)
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.eval_avg = eval_avg
        self.f_max = kwargs.get("f_max", 8000)
        self.f_min = kwargs.get("f_min", 0)
        self.hop_size = kwargs.get("hop_size", 160)
        self.mlp_ratio = mlp_ratio
        self.n_fft = kwargs.get("n_fft", 512)
        self.n_mels = kwargs.get("n_mels", 64)
        self.n_mels = kwargs.get("n_mels", 64)
        self.num_heads = num_heads
        self.outputdim = outputdim
        self.pad_last = kwargs.get("pad_last", True)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pooling = pooling
        self.qkv_bias = qkv_bias
        self.target_length = target_length
        self.win_size = kwargs.get("win_size", 512)
        self.loss = "BCE"

        if self.outputdim == 527:
            with open(os.path.join(current_dir, "class_labels_indices.csv"), "r") as f:
                self.id2label = {
                    int(line.split(",", maxsplit=3)[0]): line.split(",", maxsplit=3)[2].replace('"', "").strip("\n")
                    for line in f.readlines()[1:]
                }
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            self.id2label = None
            self.label2id = None
