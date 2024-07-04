import numpy as np
from queue import Queue
import soundcard as sc


class Recorder:
    BUFFER_SIZE = 4096

    def __init__(self, sr=48000):
        self.sr = sr
        self.audio_data_queue = Queue()
        print("Available speakers:", sc.all_speakers())
        default_speaker = sc.default_speaker()
        print("Using speaker:", default_speaker.name)

    def record(self):
        """
        开始录音，并将录音数据放入队列中，录制扬声器输出音频
        该函数含有死循环，需要在其他线程中调用
        录制结果为 float32 格式的音频数据，采样率为 self.sr，单声道
        """
        with sc.get_microphone(
            id=str(sc.default_speaker().name), include_loopback=True
        ).recorder(samplerate=self.sr, channels=1) as mic:
            while True:
                data = mic.record(Recorder.BUFFER_SIZE)
                self.audio_data_queue.put(data.reshape(-1))

    def get_queue_size(self) -> float:
        """
        获取队列中的音频数据时长，单位为秒
        """
        return self.audio_data_queue.qsize() * Recorder.BUFFER_SIZE / self.sr

    def get_queue_data(self) -> np.ndarray:
        """
        将队列中的数据包合并成一个数组并返回
        """
        all_arrays = []
        while not self.audio_data_queue.empty():
            all_arrays.append(self.audio_data_queue.get())

        # 使用 numpy.concatenate 合并所有数组
        return np.concatenate(all_arrays)
