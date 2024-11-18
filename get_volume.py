import math
import wave
import numpy as np

def calculate_db(rms, bit_width):
    reference = 2147483647.0 if bit_width == 32 else 32767.0  # 使用精确的参考值
    if rms > 0:
        db = 20 * np.log10(rms / reference)
    else:
        db = -np.inf  # 表示无信号
    return db

def calculate_rms(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

        # 根据样本宽度选择数据类型，并转换为float64来避免溢出
        if sampwidth == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        elif sampwidth == 4:
            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
        else:
            raise ValueError("Unsupported sample width: {}".format(sampwidth))

        # 如果是多通道音频，取各通道平均
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)
            audio_data = np.mean(audio_data, axis=1)

        # 定义窗口长度，计算 RMS
        k = 0.1  # 秒
        sample_rate = wf.getframerate()
        for i in range(0, n_frames, int(sample_rate * k)):
            window = audio_data[i:min(n_frames, i + int(sample_rate * k))]
            mean = np.mean(window ** 2)  # 计算平方的均值
            rms = np.sqrt(mean)  # 计算RMS
            print(
                f'{i / sample_rate:.2f}-{(i + len(window)) / sample_rate:.2f}s rms={rms:.6f} dB={calculate_db(rms, sampwidth * 8):.6f}'
            )
        return rms



# 使用示例
wav_file_path = 'me.wav'  # 替换为你的 WAV 文件路径
rms_value = calculate_rms(wav_file_path)
# print(f"当前音频的 RMS 值: {rms_value:.2f}")
