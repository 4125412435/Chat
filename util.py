import json
import os
import re
import wave

import numpy as np
import sounddevice as sd


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def list_files(dir, suffix=None):
    file_paths = []
    for file in os.listdir(dir):
        f = True
        if suffix is not None:
            f = False
            if isinstance(suffix, str):
                suffix = [suffix]
            for s in suffix:
                if file.endswith(f'.{s}'):
                    f = True
                    break
        if f:
            file_paths.append(f'{dir}/{file}')
    return file_paths


def split_text_by_language(input_text):
    """
    将输入字符串按语言标签 [ZH] 和 [JA] 分割，并提取每个块的文字内容和语言类型。

    :param input_text: 输入的字符串
    :return: 一个包含 (语言类型, 文字内容) 的列表
    """
    # 定义正则表达式，匹配 [ZH]中文[ZH] 或 [JA]日文[JA] 格式
    pattern = re.compile(r'\[(ZH|JA)\](.+?)\[\1\]')

    # 查找所有匹配项
    matches = pattern.findall(input_text)

    # 将匹配结果转换为所需格式
    result = [(lang, text) for lang, text in matches]

    return result


def play_audio(path):
    # 打开 WAV 文件
    with wave.open(path, 'rb') as wav_file:
        # 获取音频参数
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

        # 读取音频数据
        frames = wav_file.readframes(wav_file.getnframes())

        # 转换为 numpy 数组
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sample_width not in dtype_map:
            raise ValueError(f"不支持的样本宽度: {sample_width} 字节")

        dtype = dtype_map[sample_width]
        audio_data = np.frombuffer(frames, dtype=dtype)

        # 如果是多声道，将数据重塑
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)

    sd.play(audio_data, samplerate=sample_rate)
