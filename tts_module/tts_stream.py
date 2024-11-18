import threading
import queue
from time import perf_counter

import numpy
import torchaudio
import torch
import sounddevice as sd


# 示例用法
# 假设你的 TTS 对象为 tts
# player = StreamTTSPlayer(tts)
# long_text = "这是一个很长的文本，需要被分段生成和播放的例子......"
# player.text_to_speech_streaming(long_text, chunk_size=50)
# 如果需要手动停止，可以调用 player.stop()
class StreamTTSPlayer:
    def __init__(self, tts):
        self.tts = tts
        self.audio_queue = queue.Queue()
        self.sample_rate = 22050
        self.playing = threading.Event()
        self.generate_thread = None
        self.play_thread = None
        self.audio_data = numpy.array([])

    def text_to_speech_streaming_by_chunk(self, chunks, save_path, **kwargs):
        print(f"Total chunks: {len(chunks)}")

        # 初始化播放状态
        self.playing.set()

        # 启动播放线程
        # 不进行streaming播放了
        # self.play_thread = threading.Thread(target=self._play_audio)
        # self.play_thread.start()

        # 启动生成线程
        self.generate_thread = threading.Thread(target=self._generate_audio, args=(chunks, save_path), kwargs=kwargs)
        self.generate_thread.start()

    def text_to_speech_streaming(self, text, save_path, chunk_size=100, **kwargs):
        """
        启动生成和播放的线程
        """
        # 分段文本
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        self.text_to_speech_streaming_by_chunk(chunks, save_path, **kwargs)

    def _generate_audio(self, chunks, save_path, **kwargs):
        """
        按段生成语音并放入队列
        """
        chunk_speed = kwargs.get('chunk_ls', 1.0)
        if not (isinstance(chunk_speed, list) or isinstance(chunk_speed, tuple)):
            chunk_speed = [1.0] * len(chunks)
        for idx, (chunk, length_scale) in enumerate(zip(chunks, chunk_speed)):
            print(f"Generating chunk {idx + 1}/{len(chunks)}: {chunk} with length scale", length_scale)
            t1 = perf_counter()
            speech = self.tts.generate_speech(chunk, ls=length_scale)  # 生成语音
            # 保存
            self.audio_data = numpy.concatenate([self.audio_data, speech], axis=0)
            t2 = perf_counter()
            print(f"Generate speech for chunk {idx + 1} took {(t2 - t1):.2f}s")
            self.audio_queue.put(speech)  # 放入队列
        self.audio_queue.put(None)  # 生成结束标志
        print("All chunks have been generated.")
        # 保存文件
        torchaudio.save(save_path, torch.from_numpy(self.audio_data).unsqueeze(dim=0),
                        sample_rate=22050, encoding='PCM_S')

    def _play_audio(self):
        """
        从队列中获取音频并播放
        """
        while self.playing.is_set():
            speech = self.audio_queue.get()
            if speech is None:  # 遇到结束标志，停止播放
                break
            # print("Playing audio chunk...")
            # sd.play(speech, self.sample_rate)
            # sd.wait()  # 等待当前段播放完成
        print("Playback finished.")
        self.playing.clear()

    def stop(self):
        """
        停止生成和播放
        """
        self.playing.clear()
        if self.generate_thread and self.generate_thread.is_alive():
            self.generate_thread.join()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join()
