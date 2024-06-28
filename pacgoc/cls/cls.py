import os
import numpy as np
import paddle
import yaml
from paddlespeech.cli.cls import CLSExecutor
from paddle.audio.features import LogMelSpectrogram


class CLS:
    def __init__(self, model: str = "panns_cnn14", topk: int = 3):
        """
        Init model and other resources.
        """
        device = paddle.get_device()
        paddle.set_device(device)
        if device == "cpu":
            print("CLS: device is cpu")

        # 创建Infer类的实例
        self.cls = CLSExecutor()
        if hasattr(self.cls, "model"):
            return

        tag = model + "-" + "32k"  # panns_cnn14-32k
        self.cls.task_resource.set_task_model(tag, version=None)
        self.cls.cfg_path = os.path.join(
            self.cls.task_resource.res_dir, self.cls.task_resource.res_dict["cfg_path"]
        )
        self.cls.label_file = os.path.join(
            self.cls.task_resource.res_dir,
            self.cls.task_resource.res_dict["label_file"],
        )
        self.cls.ckpt_path = os.path.join(
            self.cls.task_resource.res_dir, self.cls.task_resource.res_dict["ckpt_path"]
        )

        # config
        with open(self.cls.cfg_path, "r") as f:
            self.cls._conf = yaml.safe_load(f)

        # labels
        self.cls._label_list = []
        with open(self.cls.label_file, "r") as f:
            for line in f:
                self.cls._label_list.append(line.strip())

        assert topk <= len(
            self.cls._label_list
        ), "Value of topk is larger than number of labels."
        self.topk = topk

        # model
        model_class = self.cls.task_resource.get_model_class(model)
        model_dict = paddle.load(self.cls.ckpt_path)
        self.cls.model = model_class(extract_embedding=False)
        self.cls.model.set_state_dict(model_dict)
        self.cls.model.eval()

    def preprocess(self, audio_data: np.ndarray):
        """
        Input preprocess and return paddle.Tensor stored in cls.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        feat_conf = self.cls._conf["feature"]
        waveform = audio_data.view(dtype=np.int16)
        # 将音频数据转换为float32类型
        waveform = waveform.astype(np.float32)

        # Feature extraction
        feature_extractor = LogMelSpectrogram(
            sr=feat_conf["sample_rate"],
            n_fft=feat_conf["n_fft"],
            hop_length=feat_conf["hop_length"],
            window=feat_conf["window"],
            win_length=feat_conf["window_length"],
            f_min=feat_conf["f_min"],
            f_max=feat_conf["f_max"],
            n_mels=feat_conf["n_mels"],
        )
        feats = feature_extractor(
            paddle.to_tensor(paddle.to_tensor(waveform).unsqueeze(0))
        )
        self.cls._inputs["feats"] = paddle.transpose(feats, [0, 2, 1]).unsqueeze(
            1
        )  # [B, N, T] -> [B, 1, T, N]

    def infer(self):
        """
        Model inference
        """
        self.cls.infer()

    def postprocess(self) -> list[tuple]:
        """
        Output postprocess and return human-readable results.
        Return a list of tuples, each tuple contains a label and a score.
        """
        tensor_audio = self.cls._outputs["logits"].squeeze(0).numpy()

        topk_idx = (-tensor_audio).argsort()[: self.topk]

        res = []
        for idx in topk_idx:
            label, score = self.cls._label_list[idx], tensor_audio[idx]
            res.append((label, score))
            # print(f"{label}: {score}")

        return res

    def __call__(self, audio_data: np.ndarray) -> list:
        """
        Input preprocess, model inference and output postprocess.
        """
        self.preprocess(audio_data)
        self.infer()
        return self.postprocess()
