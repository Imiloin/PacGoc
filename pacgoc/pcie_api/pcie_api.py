from . import pcie
import time
import numpy as np
from queue import Queue


class PCIe:
    PACK_HEADER = 0xAAAA  # 包头
    PACK_START = 8  # 包头长度（int16 数据个数）
    PACK_TIME = 128 / 48000  # 一个包中音频数据的时长，单位为秒
    INTERVAL = 0.002  # 循环间隔，应 < PACK_TIME - 0.0001
    TIK = INTERVAL / 10  # 接收到无效包时的等待时间，单位为秒

    def __init__(self):
        self._stop_flag = False  # 添加一个停止标志
        self._obj = pcie.PCIE()
        self.audio_data_queue = Queue()

    def stop(self):
        """
        设置停止标志，使得接收线程可以安全退出
        """
        self._stop_flag = True

    def receive(self):
        """
        2ms循环接收数据包，并将数据包中的音频数据添加到队列中
        该函数含有死循环，需要在其他线程中调用
        """
        while not self._stop_flag:
            start_time = time.time()  # 记录循环开始的时间

            self._obj.transfer()
            pack = self._obj.fetch_pack()
            pack = np.array(pack, dtype=np.uint8)

            # 将 s 重新解释为 uint16 类型的数组，注意指定小端格式
            audio_data = pack.view(np.uint16)
            if audio_data[0] == PCIe.PACK_HEADER:  # valid pack
                self.audio_data_queue.put(audio_data[PCIe.PACK_START :])
            else:  # invalid pack
                time.sleep(PCIe.TIK)  # 接收到无效包，等待一段时间
                continue

            # for sample in audio_data:
            #     print(f"{sample:04x}", end=" ")

            end_time = time.time()  # 记录循环结束的时间
            elapsed_time = end_time - start_time  # 计算循环执行所花费的时间
            if elapsed_time < PCIe.INTERVAL:
                time.sleep(PCIe.INTERVAL - elapsed_time)  # 等待直到两毫秒总时间完成

    def get_queue_size(self) -> float:
        """
        获取队列中的音频数据时长，单位为秒
        """
        return self.audio_data_queue.qsize() * PCIe.PACK_TIME

    def get_queue_data(self) -> np.ndarray:
        """
        将队列中的数据包合并成一个数组并返回
        !!使用时，需要view(dtype=np.int16)视为int16格式
        """
        all_arrays = []
        while not self.audio_data_queue.empty():
            all_arrays.append(self.audio_data_queue.get())

        # 使用 numpy.concatenate 合并所有数组
        return np.concatenate(all_arrays)
