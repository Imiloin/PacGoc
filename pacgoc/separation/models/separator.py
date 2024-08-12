import numpy as np
import os
import pickle
import soundfile as sf
import subprocess

from .utils import np_to_pytorch, evaluate_sdr, wiener, split_nparray_with_overlap

import torch
from torchlibrosa.stft import STFT, ISTFT
import pytorch_lightning as pl


# Seaparate the source from the track
class SeparatorModel(pl.LightningModule):
    def __init__(
        self,
        model,
        config,
        target_keys=["vocal", "drums", "bass", "other"],
        avg_at=None,
        using_wiener=False,
        output_wav=False,
        calc_sdr=True,
    ):
        super().__init__()
        self.model = model
        self.output_wav = output_wav
        self.calc_sdr = calc_sdr
        self.config = config
        self.target_keys = target_keys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.opt_thres = pickle.load(
            open(os.path.join(current_dir, "opt_thres.pkl"), "rb")
        )
        self.avg_at = avg_at
        self.key_dis = {}
        self.using_wiener = using_wiener
        for dickey in self.target_keys:
            self.key_dis[dickey] = np.zeros(self.config.classes_num)
        if self.config.using_whiting:
            temp = np.load("whiting_weight.npy", allow_pickle=True)
            temp = temp.item()
            self.whiting_kernel = temp["kernel"]
            self.whiting_bias = temp["bias"]
            if self.avg_at is not None:
                for dickey in self.avg_at.keys():
                    self.avg_at[dickey] = (self.avg_at[dickey] + self.whiting_bias).dot(
                        self.whiting_kernel
                    )
                    self.avg_at[dickey] = self.avg_at[dickey][0]
        window_size = 2048
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.test_step_outputs = []

        self.stft = STFT(
            n_fft=window_size,
            hop_length=config.hop_samples,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=config.hop_samples,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def get_estimate(self, batch_output_dict):
        if self.using_wiener:
            # [bs, channel, frame, freq_bin]
            return batch_output_dict["sp"].data.cpu().numpy()
        else:
            return batch_output_dict["wav"].data.cpu().numpy()

    def test_step(self, batch, batch_idx):
        """
        Args:
            batch: [1, mixture + n_target, n_samples] a numpy for one song and it sources, the first index will be the mixture
        Return:
            sdr: {target_keys: (1,) ]} sdr for each target keys
        """
        self.device_type = next(self.model.parameters()).device
        batch = batch[0].cpu().numpy()
        assert len(batch) == 1 + len(
            self.target_keys
        ), "the length of mixture + target sources should be %d" % (
            len(self.target_keys) + 1
        )
        sdr = {}
        sources = {}
        at_sources = {}
        preds = {}
        mini_batch = 8  # fixed as a small number
        segment_len = self.config.hop_samples * self.config.segment_frames
        n_samples = int(batch.shape[-1] / segment_len) * segment_len
        # resize the batch
        batch = batch[:, :n_samples]
        overlap_size = max(int(self.config.overlap_rate * segment_len), 1)
        mixture = split_nparray_with_overlap(
            batch[0], n_samples // segment_len, overlap_size
        )
        assert mixture.shape[-1] == segment_len + overlap_size, "split error"
        # get the latent embedding query
        for i, dickey in enumerate(self.target_keys):
            sources[dickey] = split_nparray_with_overlap(
                batch[i + 1, :n_samples], n_samples // segment_len, overlap_size
            )
            at_sources[dickey] = []
            sdr[dickey] = []
            if self.config.infer_type == "gd":
                for j in range(0, len(sources[dickey]), mini_batch):
                    m = sources[dickey][j : j + mini_batch]
                    output_dicts = self.model.at_model.inference(m)
                    at_vector = output_dicts["latent_output"]
                    if self.config.using_whiting:
                        at_vector = (at_vector + self.whiting_bias).dot(
                            self.whiting_kernel
                        )
                    at_sources[dickey].append(at_vector)
                at_sources[dickey] = np.concatenate(at_sources[dickey], axis=0)
                at_sources[dickey] = at_sources[dickey][:, : self.config.latent_dim]
            elif self.config.infer_type == "mean":
                at_sources[dickey] = np.zeros(
                    (len(sources[dickey]), self.config.latent_dim)
                )
                at_sources[dickey][:] = self.avg_at[dickey][: self.config.latent_dim]
            else:
                # infer via model
                raise Exception("Undefined Infer Type")
        for dickey in self.target_keys:
            preds[dickey] = []

        for i in range(0, len(mixture), mini_batch):
            for dickey in self.target_keys:
                mixture_t = np_to_pytorch(
                    mixture[i : i + mini_batch][:, :, None], self.device_type
                )
                condition_t = np_to_pytorch(
                    at_sources[dickey][i : i + mini_batch], self.device_type
                )
                batch_output_dict = self.model(mixture_t, condition_t)
                est = self.get_estimate(batch_output_dict)
                preds[dickey].append(est)

        # get stft of original audio
        if self.using_wiener:
            real_mix, img_mix = self.stft(np_to_pytorch(mixture[:], self.device_type))
            # [bs, channel, frame, bin, 2]
            mix_stft = torch.stack((real_mix, img_mix), dim=-1)
            mix_stft = mix_stft.permute(0, 2, 3, 1, 4)
            wiener_spec = []
            for dickey in self.target_keys:
                # sp if using wiener, else direct wav
                # [bs, channel, frame, bin]
                preds[dickey] = np.concatenate(preds[dickey], axis=0)
                wiener_spec.append(preds[dickey])
            # [source, bs, channel, frame, bin]
            wiener_spec = np_to_pytorch(np.array(wiener_spec), self.device_type)
            wiener_spec = wiener_spec.permute(1, 3, 4, 2, 0)
            target_stft = torch.zeros(
                mix_stft.shape + (len(self.target_keys),),
                dtype=mix_stft.dtype,
                device=mix_stft.device,
            )
            for sample in range(wiener_spec.shape[0]):
                pos = 0
                wiener_win_len = wiener_spec.shape[1]
                while pos < wiener_spec.shape[1]:
                    cur_frame = torch.arange(pos, pos + wiener_spec.shape[1])
                    pos = int(cur_frame[-1]) + 1

                    target_stft[sample, cur_frame] = wiener(
                        wiener_spec[sample, cur_frame],
                        mix_stft[sample, cur_frame],
                        1,
                        softmask=True,
                        scale_factor=8.0,
                        eps=1e-9,
                    )

            # [bs, frame, bin, channel, 2, source] -> [....]
            target_stft = target_stft.permute(4, 5, 0, 3, 1, 2).contiguous()
            real_stft = target_stft[0]
            img_stft = target_stft[1]
            for i, dickey in enumerate(self.target_keys):
                preds[dickey] = (
                    self.istft(real_stft[i], img_stft[i], segment_len)
                    .data.cpu()
                    .numpy()
                )
                if self.calc_sdr:
                    temp_sdr = evaluate_sdr(
                        ref=sources[dickey][:][:, :, None],
                        est=preds[dickey][:, :, None],
                        class_ids=np.array([0] * len(sources[dickey])),
                        mix_type="mixture",
                    )
                else:
                    temp_sdr = np.array(
                        [[0], [0], [0], [0], [0], [0]]
                    )  # blank sdr for inference
                if len(temp_sdr) >= 1:
                    sdr[dickey] = [d[0] for d in temp_sdr]
                    sdr[dickey] = np.median(sdr[dickey])
                if overlap_size == 0:
                    preds[dickey] = np.concatenate(preds[dickey], axis=0)
        else:
            for dickey in self.target_keys:
                # sp if using wiener, else direct wav
                # [bs, channel, frame, bin]
                preds[dickey] = np.concatenate(preds[dickey], axis=0)
                if self.calc_sdr:
                    temp_sdr = evaluate_sdr(
                        ref=sources[dickey][:][:, :, None],
                        est=preds[dickey],
                        class_ids=np.array([0] * len(sources[dickey])),
                        mix_type="mixture",
                    )
                else:
                    temp_sdr = np.array([[0], [0], [0], [0], [0], [0]])
                if len(temp_sdr) >= 1:
                    sdr[dickey] = [d[0] for d in temp_sdr]
                    sdr[dickey] = np.median(sdr[dickey])
                if overlap_size == 0:
                    preds[dickey] = np.concatenate(preds[dickey], axis=0)
        # output waveform
        if self.output_wav:
            # filename = str(batch_idx) + "_mixture.wav"
            filename = "mixture.wav"
            sf.write(
                os.path.join(self.config.wave_output_path, filename),
                batch[0],
                self.config.sample_rate,
            )
            for i, dickey in enumerate(self.target_keys):
                filename = self.config.output_filename
                if overlap_size > 0:
                    args = ["ffmpeg", "-y", "-loglevel", "quiet"]

                    filters = []
                    files = []

                    for j in range(len(preds[dickey])):
                        file = os.path.join(
                            self.config.wave_output_path, "chunk_{0}.wav".format(j)
                        )
                        args.extend(["-i", file])
                        files.append(file)

                        sf.write(file, preds[dickey][j], self.config.sample_rate)

                        if j < len(preds[dickey]) - 1:
                            filter_cmd = (
                                "["
                                + ("a" if j != 0 else "")
                                + "{0}][{1}]acrossfade=ns={2}:c1=tri:c2=tri".format(
                                    j, j + 1, overlap_size
                                )
                            )

                            if j != len(preds[dickey]) - 2:
                                filter_cmd += "[a{0}];".format(j + 1)

                            filters.append(filter_cmd)

                    args.extend(
                        [
                            "-filter_complex",
                            "".join(filters),
                            "-y",
                            os.path.join(self.config.wave_output_path, filename),
                        ]
                    )

                    try:
                        subprocess.check_call(args)
                    except:
                        raise "ffmpeg does not exist. Install ffmpeg or set config.overlap_rate to zero."

                    for file in files:
                        os.remove(file)
                else:
                    sf.write(
                        os.path.join(self.config.wave_output_path, filename),
                        preds[dickey],
                        self.config.sample_rate,
                    )
        self.print(batch_idx, sdr)
        self.test_step_outputs.append(sdr)
        return sdr

    def on_test_epoch_end(self):
        avg_sdr = {}
        max_sdr = {}
        min_sdr = {}
        for dickey in self.target_keys:
            q = [d[dickey] for d in self.test_step_outputs]
            q.sort()
            avg_sdr[dickey] = np.median(q)
            max_sdr[dickey] = np.max(q)
            min_sdr[dickey] = np.min(q)
        self.print("median_sdr:", avg_sdr)
        self.print("max:", max_sdr)
        self.print("min:", min_sdr)
        self.test_step_outputs.clear()  # free memory
