import logging
import sys
import threading
import time
import wave

import numpy as np
import pyaudio
import torch
import torchaudio
from PySide2.QtCore import Qt, QObject, Signal, QThread
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, \
    QMessageBox

import util
from live2d_module import Live2DWidget
from resources import resource_loader
from stt_module.whisper import Whisper
from tts_module.vits import ViTs
from ttt_module.openai_model import GPT

logging.basicConfig(level=logging.INFO)


class AsyncSpeechTask(QObject):
    finished = Signal(str)

    def __init__(self, ttt, stt, tts, message, audio):
        super(AsyncSpeechTask, self).__init__()
        self.ttt = ttt
        self.stt = stt
        self.tts = tts
        # self.tts_stream = StreamTTSPlayer(tts)
        self.message = message
        self.audio = audio

    # 异步执行
    def run(self):
        logging.info('Run async')
        if self.message:
            pass
        else:
            self.message = self.stt.translate(self.audio, origin_rate=44100, language='Chinese')[0]
            logging.info(f'translate message: {self.message}')
        response = self.ttt.ask(self.message)
        logging.info(response)

        speech_data = np.array([])
        chunks = []
        for lang, chunk in util.split_text_by_language(response):
            if lang == 'ZH':
                ls = 0.9
            else:
                ls = 1.25
            chunks.append(chunk)
            chunk = f'[{lang}]{chunk}[{lang}]'
            speech = self.tts.generate_speech(chunk, ls=ls)
            speech_data = np.concatenate([speech_data, speech], axis=0)

        torchaudio.save(resource_loader.get_path('audio', 'tmp', 'out.wav'),
                        torch.from_numpy(speech_data).unsqueeze(dim=0),
                        sample_rate=22050, encoding='PCM_S')

        # 完成任务后发送信号
        self.finished.emit(''.join(chunks))


class Live2DApp(QWidget):
    def __init__(self, args):
        super().__init__()

        self.llm = GPT(args['api_key'], 1024, model='gpt-4o-mini',
                       system_prompt=resource_loader.load_text('prompt.txt'))
        # 保存上下文
        self.llm.start_queue()
        print('loading whisper')
        self.whisper = Whisper(args['whisper_folder'], device='cuda')
        print('loading vits')
        self.vits = ViTs(resource_loader.get_path('vits', 'config', 'config.json'),
                         resource_loader.get_path('vits', 'pretrained_models', 'info.json'),
                         resource_loader.get_path('vits', 'pretrained_models'),
                         device='cuda')
        self.speaker = '天童爱丽丝'
        self.vits.load_model(self.speaker)

        self.setWindowTitle("Live2D Chat Interface")
        self.setGeometry(100, 100, 1200, 1200)

        # 主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 上半部分 - Live2D展示区域
        self.live2d_window = Live2DWidget(self, resource_loader.get_path(
            'live2d', 'hiyori_pro_en', 'runtime', 'hiyori_pro_t11.model3.json'
            # 'live2d', 'mao_pro_zh', 'runtime', 'mao_pro.model3.json'
        ))
        main_layout.addWidget(self.live2d_window)

        # 下半部分 - 文本框和录音相关区域
        self.chat_layout = QHBoxLayout()
        main_layout.addLayout(self.chat_layout)

        # 文本输入框（调大一点）
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("输入文字...")
        self.text_input.setFixedHeight(100)  # 设置高度
        self.chat_layout.addWidget(self.text_input)

        # 发送按钮
        self.send_button = QPushButton("发送", self)
        self.send_button.clicked.connect(self.send_message)
        self.chat_layout.addWidget(self.send_button)

        # 录音区域 - 垂直布局
        self.record_layout = QVBoxLayout()
        self.chat_layout.addLayout(self.record_layout)

        # 录音按钮
        self.record_button = QPushButton("录音", self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_layout.addWidget(self.record_button)

        # 录音状态标签
        self.record_status = QLabel("未录音", self)
        self.record_status.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.record_layout.addWidget(self.record_status)

        # 播放按钮
        self.play_button = QPushButton("播放录音", self)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_recording)
        self.chat_layout.addWidget(self.play_button)

        # 录音控制变量
        self.is_recording = False
        self.audio_data = None
        self.start_time = None  # 录音开始时间

    def on_speech_generate_finish(self, result):
        print('Finish generate speech')
        util.play_audio(resource_loader.get_path('audio', 'tmp', 'out.wav'))
        self.live2d_window.speech(result, resource_loader.get_path('audio', 'tmp', 'out.wav'))

    def send_message(self):
        # 发送按钮点击后执行的函数
        if self.text_input.text() or self.audio_data is not None:
            self.speech_task = AsyncSpeechTask(self.llm, self.whisper, self.vits, self.text_input.text(),
                                               None if self.audio_data is None else self.audio_data.copy())
            self.speech_task.finished.connect(self.on_speech_generate_finish)
            self.thread = QThread(self)
            self.speech_task.moveToThread(self.thread)
            self.thread.started.connect(self.speech_task.run)
            self.thread.start()

            self.text_input.clear()
            self.audio_data = None
            self.record_status.setText('未录音')
            self.play_button.setEnabled(False)

    def toggle_recording(self):
        if not self.is_recording:
            # 开始录音
            self.is_recording = True
            self.audio_data = []
            self.record_button.setText("停止录音")
            self.start_time = time.time()  # 开始时间
            self.record_status.setText("录音中...")
            threading.Thread(target=self.record_audio).start()
        else:
            # 停止录音
            self.is_recording = False
            self.record_button.setText("录音")
            self.save_recording()

    def record_audio(self):
        # 初始化录音参数
        chunk = 1024  # 每次读取的帧数
        format = pyaudio.paInt16  # 16位深度，PCM格式
        channels = 1  # 单声道
        rate = 44100  # 采样率

        p = pyaudio.PyAudio()
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        self.audio_data = []  # 用于存储音频数据

        while self.is_recording:
            data = stream.read(chunk)
            self.audio_data.append(data)

        # 停止流并关闭
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 将数据转换为PCM-S格式的numpy数组
        self.audio_data = np.frombuffer(b''.join(self.audio_data), dtype=np.int16).astype(np.float32)

    def save_recording(self):
        # 保存音频文件
        wf = wave.open(resource_loader.get_path('audio', 'tmp', 'recording.wav'), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_data))
        wf.close()

        # 录音完成状态更新
        duration = time.time() - self.start_time  # 计算时长
        self.record_status.setText(f"录音完成，时长: {duration:.2f} 秒")
        self.play_button.setEnabled(True)  # 启用播放按钮
        QMessageBox.information(self, "提示", "录音已保存为 recording.wav")

    def play_recording(self):
        # 播放录音
        wf = wave.open(resource_loader.get_path('audio', 'tmp', 'recording.wav'), 'rb')
        p = pyaudio.PyAudio()

        # 打开音频流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    config = util.load_config('config.json')
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_loader.get_path('image', 'logo.png')))
    window = Live2DApp(config)
    window.show()
    sys.exit(app.exec_())
