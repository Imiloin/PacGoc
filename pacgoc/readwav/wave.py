import os
import wave
import numpy as np
import time
from queue import Queue


class Wave:
    def __init__(self, file_path: os.PathLike, chunk_size: int = 1024):
        self.wav_file = file_path
        self.chunk_size = chunk_size
        self.audio_data_queue = Queue()

        with wave.open(self.wav_file, "rb") as wav_file:
            self.n_channels = wav_file.getnchannels()
            self.sample_width = wav_file.getsampwidth()
            self.frame_rate = wav_file.getframerate()
            self.n_frames = wav_file.getnframes()
            self.comp_type = wav_file.getcomptype()
            self.comp_name = wav_file.getcompname()

        print(
            f"通道数: {self.n_channels}, 采样宽度: {self.sample_width}字节, 帧率: {self.frame_rate}帧/秒, 总帧数: {self.n_frames}"
        )
        print(f"压缩类型: {self.comp_type}, 压缩名称: {self.comp_name}")

    def get_sample_rate(self) -> int:
        return self.frame_rate

    def _get_dtype(self, sample_width: int) -> np.dtype:
        if sample_width == 1:
            print("Warning: 8位音频数据，建议使用16位")
            return np.uint8
        elif sample_width == 2:
            return np.int16
        elif sample_width == 4:
            print("Warning: 32位音频数据，建议使用16位")
            return np.int32
        else:
            raise ValueError("Unsupported sample width")

    def read(self):
        """
        读取音频数据并放入队列中
        该函数中含死循环，直到读取完整个音频文件
        """
        with wave.open(self.wav_file, "rb") as wav_file:
            dtype = self._get_dtype(self.sample_width)
            sleep_time = self.chunk_size / self.frame_rate
            while True:
                frames = wav_file.readframes(self.chunk_size)
                if not frames:
                    break
                data = np.frombuffer(frames, dtype=dtype)
                # 重塑数组，使其第二维对应于通道
                data = data.reshape(-1, self.n_channels)
                # 仅提取单通道的数据
                channel_data = data[:, 0]
                self.audio_data_queue.put(channel_data)
                time.sleep(sleep_time)

    def get_queue_size(self) -> float:
        """
        获取队列中的音频数据时长，单位为秒
        """
        return self.audio_data_queue.qsize() * self.chunk_size / self.frame_rate

    def get_queue_data(self) -> np.ndarray:
        """
        将队列中的数据包合并成一个数组并返回
        """
        all_arrays = []
        while not self.audio_data_queue.empty():
            all_arrays.append(self.audio_data_queue.get())

        # 使用 numpy.concatenate 合并所有数组
        combined_data = np.concatenate(all_arrays)
        if self.sample_width != 2:
            combined_data = combined_data.astype(np.int16)
        return combined_data
