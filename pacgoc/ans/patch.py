from typing import Any, Dict
import librosa
import io
import numpy as np
import soundfile as sf
import torch
from modelscope.fileio import File
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input
from modelscope.utils.audio.audio_utils import audio_norm

def custom_preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
    if self.stream_mode:
        raise TypeError('This model does not support stream mode!')
    # add ndarray support
    if isinstance(inputs, np.ndarray):
        data1 = inputs
        fs = 16000 ######
    elif isinstance(inputs, bytes):
        data1, fs = sf.read(io.BytesIO(inputs))
    elif isinstance(inputs, str):
        file_bytes = File.read(inputs)
        data1, fs = sf.read(io.BytesIO(file_bytes))
    else:
        raise TypeError(f'Unsupported type {type(inputs)}.')
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if fs != self.SAMPLE_RATE:
        data1 = librosa.resample(
            data1, orig_sr=fs, target_sr=self.SAMPLE_RATE)
    data1 = audio_norm(data1)
    data = data1.astype(np.float32)
    self.custom_max_orig = np.max(np.abs(data))
    inputs = np.reshape(data, [1, data.shape[0]])
    return {'ndarray': inputs, 'nsamples': data.shape[0]}

def custom_forward(self, inputs: Dict[str, Any],
            **forward_params) -> Dict[str, Any]:
    ndarray = inputs['ndarray']
    if isinstance(ndarray, torch.Tensor):
        ndarray = ndarray.cpu().numpy()
    nsamples = inputs['nsamples']
    decode_do_segement = False
    window = 16000
    stride = int(window * 0.75)
    print('inputs:{}'.format(ndarray.shape))
    b, t = ndarray.shape  # size()
    if t > window * 120:
        decode_do_segement = True

    if t < window:
        ndarray = np.concatenate(
            [ndarray, np.zeros((ndarray.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        print('padding: {}'.format(padding))
        ndarray = np.concatenate(
            [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            print('padding: {}'.format(padding))
            ndarray = np.concatenate(
                [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
    print('inputs after padding:{}'.format(ndarray.shape))
    with torch.no_grad():
        ndarray = torch.from_numpy(np.float32(ndarray)).to(self.device)
        b, t = ndarray.shape
        if decode_do_segement:
            outputs = np.zeros(t)
            give_up_length = (window - stride) // 2
            current_idx = 0
            while current_idx + window <= t:
                print('current_idx: {}'.format(current_idx))
                tmp_input = dict(noisy=ndarray[:, current_idx:current_idx
                                                + window])
                tmp_output = self.model(
                    tmp_input, )['wav_l2'][0].cpu().numpy()
                end_index = current_idx + window - give_up_length
                if current_idx == 0:
                    outputs[current_idx:
                            end_index] = tmp_output[:-give_up_length]
                else:
                    outputs[current_idx
                            + give_up_length:end_index] = tmp_output[
                                give_up_length:-give_up_length]
                current_idx += stride
        else:
            outputs = self.model(
                dict(noisy=ndarray))['wav_l2'][0].cpu().numpy()
    # scale data to get largest amplitude to be 32768
    self.custom_max_out = np.max(np.abs(outputs))
    if hasattr(self, 'custom_max_orig'):
        if self.custom_max_out > self.custom_max_orig * 0.5:
            # outputs = outputs * self.custom_max_orig / self.custom_max_out
            outputs = outputs / self.custom_max_out
    # convert to 16-bit PCM
    outputs = (outputs[:nsamples] * 32768).astype(np.int16).tobytes()
    return {OutputKeys.OUTPUT_PCM: outputs}

def custom_postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    input = inputs[OutputKeys.OUTPUT_PCM]
    if 'output_path' in kwargs.keys():
        sf.write(
            kwargs['output_path'],
            np.frombuffer(input, dtype=np.int16),
            self.SAMPLE_RATE)
    return inputs
